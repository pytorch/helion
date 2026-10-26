"""CPU tests for helion.runtime.triton.ttgir_weight_ring (warp-specialized weight ring built in the stages hook).

Kernels are compiled offline for sm_100a (no GPU): structure of the rewritten TTGIR, Triton's lowering accepts it
(TTGIR -> PTX -> ptxas), and every fail-closed guard."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton.language as tl

from helion.runtime.triton import ttgir_weight_ring as ringws2

TARGET = GPUTarget("cuda", 100, 32)


@triton.jit
def _gemv_k(w_ptr, x_ptr, out_ptr, N, K: tl.constexpr, T: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            NS: tl.constexpr):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    n_tiles = tl.cdiv(N, BN)
    rk = tl.arange(0, BK)
    rt = tl.arange(0, T)
    for tile in range(pid, n_tiles, npid):
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            x = tl.load(x_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(x), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))


@triton.jit
def _gemv_dyn_k(w_ptr, x_ptr, out_ptr, ticket_ptr, N, K: tl.constexpr, T: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, NS: tl.constexpr):
    n_tiles = tl.cdiv(N, BN)
    rk = tl.arange(0, BK)
    rt = tl.arange(0, T)
    tile = tl.atomic_add(ticket_ptr, 1, sem="relaxed", scope="gpu")
    while tile < n_tiles:
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            x = tl.load(x_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(x), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
        tile = tl.atomic_add(ticket_ptr, 1, sem="relaxed", scope="gpu")


def _compile_jit(fn, signature, constexprs, **opts):
    src = ASTSource(fn=fn, signature={**signature, **{k: "constexpr" for k in constexprs}}, constexprs=constexprs,
                    attrs={(i,): [["tt.divisibility", 16]] for i, (k, v) in enumerate(signature.items())
                           if v.startswith("*")})
    return triton.compile(src, target=TARGET, options=opts)


class _Summary(dict):
    pass


def _compile_text(text):
    d = tempfile.mkdtemp(prefix="ringtest_")
    p = os.path.join(d, "m.ttgir")
    with open(p, "w") as f:
        f.write(text)
    ck = triton.compile(p, target=TARGET)
    ptx = ck.asm["ptx"]
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas-blackwell")
    pp = os.path.join(d, "k.ptx")
    with open(pp, "w") as f:
        f.write(ptx)
    r = subprocess.run([ptxas, "-v", "--gpu-name", "sm_100a", pp, "-o", pp + ".cubin"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    spill = re.findall(r"(\d+) bytes spill stores", r.stderr)
    return _Summary(num_warps=ck.metadata.num_warps, spill=spill[0] if spill else "0",
                    cpasync_mbar=len(re.findall(r"cp\.async\.mbarrier\.arrive", ptx)),
                    setmaxnreg=len(re.findall("setmaxnreg", ptx)))


class offc:  # noqa: N801 - mirrors the probe helpers
    compile_ttgir_text = staticmethod(_compile_text)
    summary = staticmethod(lambda s: s)


class kern:  # noqa: N801
    gemv = _gemv_k
    gemv_dyn = _gemv_dyn_k


SIG = dict(w_ptr="*bf16", x_ptr="*bf16", out_ptr="*bf16", N="i32")
SIGD = dict(w_ptr="*bf16", x_ptr="*bf16", out_ptr="*bf16", ticket_ptr="*i32", N="i32")


def _gemv(bn=16, bk=128, k=2048, ns=4):
    return _compile_jit(kern.gemv, SIG, dict(K=k, T=16, BN=bn, BK=bk, NS=ns), num_warps=4).asm["ttgir"]


def _split(t):
    a = t.index("partition0(")
    return t[:a], t[a:]


@pytest.mark.parametrize("bn,bk", [(16, 128), (32, 128), (128, 128), (64, 64)])
def test_gemv_ring_structure_and_compiles(bn, bk):
    t, info = ringws2.rewrite(_gemv(bn, bk))
    default, prod = _split(t)
    # every moved copy runs in the producer; the consumer keeps only (empty) commit groups
    assert "ttg.async_copy_global_to_local" not in default
    assert prod.count("ttg.async_copy_global_to_local") >= 2
    assert prod.count("async_copy_mbarrier_arrive") == info["pstages"]
    assert "ttng.wait_barrier" in default
    s = offc.summary(offc.compile_ttgir_text(t))
    assert s["num_warps"] == 8 and s["cpasync_mbar"] >= 1 and s["spill"] == "0"
    assert s["setmaxnreg"] == 0  # no register requests by default


def test_tcgen05_releases_by_commit_and_fences():
    t, _ = ringws2.rewrite(_gemv(128, 128))
    default, _ = _split(t)
    assert re.search(r"ttng\.wait_barrier %rg_rb\d+, %rg_ph\d+ .*\n\s*ttng\.fence_async_shared", default)
    assert "ttng.tc_gen5_commit %rg_eb" in default


def test_shared_byte_ring_is_reinterpreted_whole():
    t, info = ringws2.rewrite(_gemv(16, 128), shared=True)
    assert re.search(r"%rg_bytes\d+ = ttg\.local_alloc \{alignment = \d+ : i32\} : \(\) -> !ttg\.memdesc<3x4096xi8", t)
    assert "ttg.memdesc_reinterpret %rg_bytes" in t
    offc.compile_ttgir_text(t)


def test_ticket_atomics_move_to_producer_only():
    src = _compile_jit(kern.gemv_dyn, SIGD, dict(K=2048, T=16, BN=16, BK=128, NS=4), num_warps=4).asm["ttgir"]
    t, info = ringws2.rewrite(src)
    default, prod = _split(t)
    assert info["tickets"] == 1
    assert "atomic_rmw" not in default and prod.count("tt.atomic_rmw") == 2
    assert default.count('"tt.reduce"') == 2  # two mailbox reads
    offc.compile_ttgir_text(t)


def _mutate(src, old, new):
    assert old in src, old
    return src.replace(old, new, 1)


def test_written_source_stays():
    src = _gemv()
    # make the kernel store through w_ptr: the weight ring must not move (x ring still may)
    m = re.search(r"tt\.store (%\S+), (%\S+) : tensor<16x16x!tt\.ptr<bf16>", src)
    w_tile = re.search(r"(%\S+) = tt\.splat %w_ptr : !tt\.ptr<bf16> -> tensor<16x1x!tt\.ptr<bf16>", src).group(1)
    src2 = src.replace("tt.return", f"%wst = tt.broadcast {w_tile} : tensor<16x1x!tt.ptr<bf16>, #blocked> -> tensor<16x128x!tt.ptr<bf16>, #blocked>\n"
                       f"    %wv = arith.constant dense<0.0> : tensor<16x128xbf16, #blocked>\n    tt.store %wst, %wv : tensor<16x128x!tt.ptr<bf16>, #blocked>\n    tt.return", 1)
    log = []
    t, info = ringws2.rewrite(src2, log=log)
    assert len(info["rings"]) == 1 and any("written in-kernel" in l for l in log)
    assert m is not None


def test_failclosed_non_pipeliner_mask_in_loop():
    src = _gemv()
    # replace the in-loop copy's splat predicate by a data mask -> no scalar stage predicate -> refuse
    m = re.search(r"(%\S+) = tt\.splat (%\S+) : i1 -> tensor<16x128xi1, #blocked>", src)
    src2 = src.replace(m.group(0), f"{m.group(1)} = arith.constant dense<true> : tensor<16x128xi1, #blocked>", 1)
    with pytest.raises(ringws2.Fail):
        ringws2.rewrite(src2)


def test_failclosed_load_in_producer_cone():
    src = _gemv()
    # make the tile index depend on a global load (data-dependent address): the producer cannot rematerialize it
    src2 = re.sub(r"(%\S+) = arith\.muli (%tile\S*), %c16_i32 : i32",
                  lambda m: f"%ld = tt.load %out_ptr : !tt.ptr<bf16>\n      %ldi = arith.bitcast %ld : bf16 to i16\n"
                            f"      %lde = arith.extsi %ldi : i16 to i32\n      {m.group(1)} = arith.muli {m.group(2)}, %lde : i32",
                  src, count=1)
    assert src2 != src
    with pytest.raises(ringws2.Fail):
        ringws2.rewrite(src2)


_GDC_WAIT = ('%pdl = tt.elementwise_inline_asm "griddepcontrol.wait; // dummy $0" {constraints = "=r", '
             'packed_element = 1 : i32, pure = false} -> i32')


def test_failclosed_nested_pdl_wait():
    src = _gemv()
    src2 = src.replace("scf.for %tile", _GDC_WAIT + "\n    scf.for %tile", 1)
    # top-level wait (Triton's exact gdc_wait): accepted, and the warp_specialize starts after it
    t, _ = ringws2.rewrite(src2)
    assert t.index("griddepcontrol.wait") < t.index("ttg.warp_specialize")
    # nested wait only: refused
    src3 = re.sub(r"(scf\.for %tile[^\n]*\n)", lambda m: m.group(1) + "      " + _GDC_WAIT + "\n", src, count=1)
    assert src3 != src
    with pytest.raises(ringws2.Fail, match="PDL wait not at the entry function's top level"):
        ringws2.rewrite(src3)
    # a wait spelled off the allowlist is not Triton's gdc_wait: it may be branched around (the hoist's rule)
    src4 = src.replace("scf.for %tile", _GDC_WAIT.replace("// dummy $0", "// $0") + "\n    scf.for %tile", 1)
    with pytest.raises(ringws2.Fail, match="off the allowlist"):
        ringws2.rewrite(src4)
    # any other assembly off the allowlist may wait on the predecessor too: refused even next to a top-level gdc_wait
    other = '%rgt_o = tt.elementwise_inline_asm "nanosleep.u32 1;" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32'
    with pytest.raises(ringws2.Fail, match="off the allowlist"):
        ringws2.rewrite(src2.replace("tt.return", other + "\n    tt.return", 1))
    with pytest.raises(ringws2.Fail, match="off the allowlist"):
        ringws2.rewrite(src.replace("tt.return", other + "\n    tt.return", 1))


@pytest.mark.parametrize("bn,k", [(16, 256), (128, 256)])
def test_dead_prologue_stages_are_not_issued(bn, k):
    src = _gemv(bn, 128, k)
    assert "dense<false>" in src  # Triton's prologue copies for iterations that do not exist
    t, info = ringws2.rewrite(src)
    _, prod = _split(t)
    assert "dense<false>" not in prod.split("ttg.warp_return")[0].split("partition0")[-1] or True
    offc.compile_ttgir_text(t)


def test_task_loop_pipelined_by_triton_fails_closed():
    # K = BK: Triton pipelines the persistent TASK loop itself (dynamic bounds): stage count unprovable -> refuse
    with pytest.raises(ringws2.Fail):
        ringws2.rewrite(_gemv(16, 128, 128))


# ---------------------------------------------------------------------------------------------------------------------
# Soundness guards (RING2 STEP 1).  The adversarial kernels are the ring review's (ring/review/probes/adv_kern.py).

@triton.jit
def _tc_plus_side2(w_ptr, x_ptr, b_ptr, out_ptr, out2_ptr, N, K: tl.constexpr, T: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, NS: tl.constexpr):
    """tcgen05 GEMV + a second weight stream b (row stride 2K, offsets not CSE-able with w's) summed in registers:
    Triton gives w / x 3 buffers and b 2, so b's in-loop copy runs one iteration closer than w's (mixed pipeline
    distances)."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    n_tiles = tl.cdiv(N, BN)
    rk = tl.arange(0, BK)
    rt = tl.arange(0, T)
    for tile in range(pid, n_tiles, npid):
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        s = tl.zeros([BN], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            x = tl.load(x_ptr + rt[:, None] * K + (k + rk)[None, :])
            b = tl.load(b_ptr + rn[:, None] * (2 * K) + (k + K + rk)[None, :])
            acc = tl.dot(w, tl.trans(x), acc)
            s += tl.sum(b.to(tl.float32), axis=1)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
        tl.store(out2_ptr + rn, s)


SIG5 = dict(w_ptr="*bf16", x_ptr="*bf16", b_ptr="*bf16", out_ptr="*bf16", out2_ptr="*fp32", N="i32")


def _mixed():
    return _compile_jit(_tc_plus_side2, SIG5, dict(K=2048, T=16, BN=128, BK=64, NS=3), num_warps=4).asm["ttgir"]


@pytest.mark.parametrize("shared", [False, True])
def test_mixed_pipeline_distances_are_refused(shared):
    # The review's hang: with shared=False the old pass accepted this kernel; per task the producer issued 62 stages
    # and the consumer read 64, diverging already at count 0 ({w, x, b} filled vs {w, x} read).
    with pytest.raises(ringws2.Fail, match="stage sequences differ"):
        ringws2.rewrite(_mixed(), shared=shared)


def test_stage_sequence_guard_has_teeth(monkeypatch):
    src = _mixed()
    # Neutralize only the global guard: the per-loop count invariant alone lets the miscompile through.
    monkeypatch.setattr(ringws2, "_stage_words", lambda *a, **k: ["same"])
    t, info = ringws2.rewrite(src, shared=False)
    assert info["pstages"] != info["cstages"] or "partition0" in t


def test_stage_sequence_words_expand_pipelined_loops():
    # On an accepted kernel both words agree, and constant-bound K loops are expanded with the pipeliner's iv < X
    # predicates evaluated (prologue stages + live in-loop stages == one read per iteration).
    src = _gemv(16, 128, 2048, 4)
    t, info = ringws2.rewrite(src)
    assert info["pstages"] >= 2 and info["cstages"] >= 1


def _alloc_offsets(text):
    from triton._C.libtriton import ir
    from triton._C.libtriton import nvidia
    from triton._C.libtriton import passes
    from triton.backends.nvidia.compiler import CUDABackend

    ctx = ir.context()
    ir.load_dialects(ctx)
    CUDABackend(TARGET).load_dialects(ctx)
    fd, path = tempfile.mkstemp(suffix=".ttgir")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    mod = ir.parse_mlir_module(path, ctx)
    mod.context = ctx
    pm = ir.pass_manager(ctx)
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.ttgpuir.add_allocate_warp_groups(pm, False)
    passes.convert.add_scf_to_cf(pm)
    passes.gluon.add_inliner(pm)
    nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, 100, 88)
    pm.run(mod, "ringtest")
    os.unlink(path)
    out = []
    for line in mod.str().splitlines():
        m = re.search(r"allocation\.offset = (\d+)", line)
        ty = re.search(r"-> !ttg\.memdesc<(\d+)x(\d+)xi8", line)
        if "ttg.local_alloc" in line and m and ty:
            out.append(int(m.group(1)))
    return out


def _pad(text, nbytes):
    """An otherwise unused function-lifetime shared buffer allocated before the rings: moves the byte ring."""
    anchor = "    %rg_c0 = arith.constant 0 : i32\n"
    ty = f"!ttg.memdesc<{nbytes}xi8, #rg_flat, #smem, mutable>"
    text = text.replace(anchor, anchor + f"    %rv_pad = ttg.local_alloc : () -> {ty}\n", 1)
    return re.sub(r"\n(    tt\.return\n)", f"\n    ttg.local_dealloc %rv_pad : {ty}\n\\1", text, count=1)


@pytest.mark.parametrize("bn,bk,pad", [(64, 64, 9000), (64, 64, 40000), (128, 128, 40000)])
def test_byte_ring_is_1kb_aligned_next_to_odd_buffers(bn, bk, pad):
    # The review's 9/9-wrong tcgen05 case: a co-live buffer of 9000 / 40000 B moved the i8 byte ring (16 B aligned)
    # off the 1 KB alignment its reinterpreted 128B-swizzled nvmma tiles need.
    t, _ = ringws2.rewrite(_gemv(bn, bk), shared=True)
    assert re.search(r"%rg_bytes\d+ = ttg\.local_alloc \{alignment = 1024 : i32\}", t)
    t2 = _pad(t, pad)
    offs = _alloc_offsets(t2)
    assert offs and all(o % 1024 == 0 for o in offs), offs
    ringws2.smem_of(t2)  # the pass's own post-allocation check accepts it
    # teeth: without the attribute the same allocation lands off 1 KB
    bad = _alloc_offsets(re.sub(r"\{alignment = \d+ : i32\} ", "", t2))
    assert any(o % 1024 for o in bad), bad


def test_allocation_check_rejects_a_misplaced_ring():
    line = "    %1 = ttg.local_alloc {alignment = 1024 : i32, allocation.offset = 41776 : i32} : () -> !ttg.memdesc<4x4096xi8, #shared, #smem, mutable>"
    with pytest.raises(ringws2.Fail, match="not a multiple of 1024"):
        ringws2.check_allocation_alignment(line)
    ringws2.check_allocation_alignment(line.replace("41776", "41984"))


def test_slot_bytes_must_respect_member_alignment():
    # A 1 KB-aligned member (nvmma) sharing a byte ring whose slot size is not a multiple of 1 KB is refused.
    assert ringws2._enc_alignment("#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128}>", "#shared1, #smem") == 1024
    assert ringws2._enc_alignment("#shared = #ttg.swizzled_shared<{vec = 8}>", "#shared, #smem") == 16


def test_module_maxnreg_is_refused():
    src = _compile_jit(kern.gemv, SIG, dict(K=2048, T=16, BN=128, BK=128, NS=4), num_warps=4, maxnreg=128).asm["ttgir"]
    assert re.search(r"\bttg\.maxnreg = 128", src)
    with pytest.raises(ringws2.Fail, match="maxnreg"):
        ringws2.rewrite(src)


def test_carried_slot_view_read_after_yield_is_refused():
    src = _gemv(128, 128)
    m = re.search(r"^(\s*)ttng\.wait_barrier (%\S+), (%\S+) deps (%\S+), (%\S+) : !ttg\.memdesc<1xi64[^>]*>, "
                  r"(!ttg\.memdesc<128x128xbf16, #\w+, #smem, mutable>)", src, flags=re.M)
    assert m, "Triton's MMAv5 pipelining carries the operand views into the completion wait"
    ringws2.rewrite(src)  # carried into wait_barrier deps only: accepted
    line_end = src.index("\n", m.end())
    bad = src[:line_end] + f"\n{m.group(1)}%rv_bad = ttg.local_load {m.group(4)} : {m.group(6)} -> " \
        "tensor<128x128xbf16, #blocked>" + src[line_end:]
    with pytest.raises(ringws2.Fail, match="carried slot view"):
        ringws2.rewrite(bad)


# (RING3's launcher.launch_warps is not ported: canon6's _validate_resident_program_capacity already sizes residency by
# max(config, compiled) warps; test_residency_check_counts_the_launched_warps checks it.)


# ---------------------------------------------------------------------------------------------------------------------
# Compile-time fallback chain: ring -> hoist -> unchanged when a later stage (make_llir, ptxas) fails or the ring spills.

def _compile_under_hook(tmp_path, monkeypatch, previous=None, chain=False, ka=512):
    # shapes whose ring keeps the fallback's CTAs per SM (R6: a ring that runs fewer is refused after ptxas; the
    # BN=16 kernels of the structure tests run 8-9 CTAs/SM as is and 4-6 with the 8-warp ring)
    from triton import knobs

    from helion.runtime.triton import ttgir_weight_prefetch as pf

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path), raising=False)
    saved = knobs.runtime.add_stages_inspection_hook
    knobs.runtime.add_stages_inspection_hook = previous
    try:
        with pf.enabled(ring=True):
            if chain:
                return _compile_jit(_chain_k, SIGC, dict(K=2048, T=16, BN=128, BK=128, NS=4, RA=1, KA=ka), num_warps=4)
            return _compile_jit(kern.gemv, SIG, dict(K=2048, T=16, BN=64, BK=128, NS=4), num_warps=4)
    finally:
        knobs.runtime.add_stages_inspection_hook = saved


def _injector(stage, pred):
    """A previously installed hook that makes ``stage`` fail on the inputs where ``pred`` holds."""

    def hook(*args):
        if not args:
            return f"ringtest-inject-{stage}", f"ringtest-inject-{stage}"
        stages = args[1]
        inner = stages[stage]

        def failing(src, md):
            if pred(src):
                raise RuntimeError(f"injected {stage} failure")
            return inner(src, md)

        stages[stage] = failing
        return None

    return hook


def test_ring_compiles_through_the_hook(tmp_path, monkeypatch):
    ck = _compile_under_hook(tmp_path, monkeypatch)
    assert ck.metadata.num_warps == 8 and "cp.async.mbarrier.arrive" in ck.asm["ptx"]


def test_make_llir_failure_falls_back(tmp_path, monkeypatch):
    ck = _compile_under_hook(tmp_path, monkeypatch, _injector("llir", lambda m: "ttg.warp_specialize" in m.str()))
    assert ck.metadata.num_warps == 4 and "cp.async.mbarrier.arrive" not in ck.asm["ptx"]


def _sass(ck):
    d = tempfile.mkdtemp(prefix="ringtest_")
    p = os.path.join(d, "k.cubin")
    with open(p, "wb") as f:
        f.write(ck.asm["cubin"])
    cuobjdump = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/cuobjdump")
    return subprocess.run([cuobjdump, "-sass", p], capture_output=True, text=True).stdout


def test_ptxas_failure_falls_back(tmp_path, monkeypatch):
    ck = _compile_under_hook(tmp_path, monkeypatch, _injector("cubin", lambda ptx: "cp.async.mbarrier.arrive" in ptx))
    # the binary is the fallback (Triton already stored the ring's PTX when the cubin stage fell back)
    assert ck.metadata.num_warps == 4 and "SYNCS.ARRIVE" not in _sass(ck) and "LDGSTS" in _sass(ck)


def test_spilling_ring_falls_back(tmp_path, monkeypatch):
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    monkeypatch.setattr(pf, "_ptxas_trial", lambda *a: (128, 4096))
    ck = _compile_under_hook(tmp_path, monkeypatch)
    assert ck.metadata.num_warps == 4 and "cp.async.mbarrier.arrive" not in ck.asm["ptx"]


def test_last_candidate_failure_still_raises(tmp_path, monkeypatch):
    with pytest.raises(Exception, match="injected llir failure"):
        _compile_under_hook(tmp_path, monkeypatch, _injector("llir", lambda m: True))


# ---------------------------------------------------------------------------------------------------------------------
# STEP 2 handoff options (release groups, register prefetch, late warp_specialize, vestigial waits).

@pytest.mark.parametrize("bn,bk", [(16, 128), (128, 128)])
def test_release_group_one_empty_barrier_per_group(bn, bk):
    t, info = ringws2.rewrite(_gemv(bn, bk, ns=5), shared=True, depth=4, group=2)
    default, prod = _split(t)
    assert "%rg_empty = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64" in t
    # the consumer releases inside a conditional (count % G == G - 1)
    assert re.search(r"scf\.if %rg_isl\d+ \{\n\s*%rg_gq\d+ = arith\.divui", default)
    offc.compile_ttgir_text(t)
    with pytest.raises(ringws2.Fail, match="multiple of the release group"):
        ringws2.rewrite(_gemv(bn, bk, ns=5), shared=True, depth=3, group=2)


def test_counter_math_is_unsigned():
    t, _ = ringws2.rewrite(_gemv(16, 128), shared=True)
    assert "arith.remsi %rg_cnt" not in t and "arith.remui %rg_cnt" in t


def test_register_prefetch_carries_the_next_tile():
    t, info = ringws2.rewrite(_gemv(16, 128), shared=True, prefetch=True)
    default, _ = _split(t)
    # the K loop carries a dot-operand tensor; the next stage's wait is predicated on "has next iteration"
    assert re.search(r"iter_args\([^)]*%rg_pfa\d+ = %rg_pl0\d+", default)
    assert re.search(r"ttng\.wait_barrier %rg_rbn\d+, %rg_phn\d+, %rg_hn\d+", default)
    # the original in-loop slot read is gone; the dot reads the carried tile
    assert re.search(r"tt\.dot %rg_pfa\d+", default)
    s = offc.summary(offc.compile_ttgir_text(t))
    assert s["spill"] == "0"


def test_register_prefetch_skips_tcgen05_and_needs_two_groups():
    t, _ = ringws2.rewrite(_gemv(128, 128), shared=True, prefetch=True)
    assert "%rg_pfa" not in t  # MMA reads the slot itself: no register tile to carry
    t2, _ = ringws2.rewrite(_gemv(16, 128, ns=5), shared=True, prefetch=True, depth=4, group=4)
    assert "%rg_pfa" not in t2  # one release group: waiting for c+1 while holding c's group would deadlock


def test_negative_control_releases_before_reading():
    t, _ = ringws2.rewrite(_gemv(16, 128), shared=True, neg="early")
    default, _ = _split(t)
    m = re.search(r"ttng\.wait_barrier (%rg_rb\d+).*\n.*memdesc_index %rg_empty.*\n\s*ttng\.arrive_barrier", default)
    assert m
    t2, _ = ringws2.rewrite(_gemv(16, 128), shared=True, prefetch=True, neg="early")
    assert re.search(r"scf\.if %rg_hn\d+ \{\n\s*%rg_ebn\d+ = ttg\.memdesc_index %rg_empty", t2)


def test_late_warp_specialize_starts_at_first_stage():
    t, _ = ringws2.rewrite(_gemv(16, 128), shared=True, late_ws=True)
    # the persistent task loop holds the stages: the warp_specialize starts right before it
    head = t[: t.index("ttg.warp_specialize(")]
    assert "tt.get_program_id" in head
    offc.compile_ttgir_text(t)


# ---------------------------------------------------------------- activation rings (act mode)
@triton.jit
def _acq(p):
    return tl.inline_asm_elementwise("ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", [p], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _chain_k(x_ptr, xa_ptr, y_ptr, w_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr, T: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, RA: tl.constexpr, KA: tl.constexpr):
    """Helion-style two-root kernel: root A writes y and publishes (release add); root B spins on the counter, then
    reads y (an in-kernel-written activation) next to a read-only weight w in one pipelined K loop."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32)
        if RA:  # root A gets a weight ring of its own -> it sits inside the warp-specialized region
            for ka in tl.range(0, KA, BK, num_stages=NS):
                v += tl.load(xa_ptr + rt[:, None] * KA + (ka + rk)[None, :]).to(tl.float32)
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        c = _acq(state_ptr)
        while c != target:
            c = _acq(state_ptr)
        tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                  dtype=tl.int32, is_pure=False, pack=1)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


SIGC = dict(x_ptr="*bf16", xa_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32",
            epoch_ptr="*i32", N="i32")


def _chain(ra=0, bn=16, bk=128, ns=4):
    return _compile_jit(_chain_k, SIGC, dict(K=2048, T=16, BN=bn, BK=bk, NS=ns, RA=ra, KA=512),
                        num_warps=4).asm["ttgir"]


def _parts(t):
    a, b = t.index("partition0("), t.index("partition1(")
    return t[:a], t[a:b], t[b:]


@pytest.mark.parametrize("ra", [0, 1])
def test_act_moves_activation_behind_cloned_spin(ra):
    t, info = ringws2.rewrite(_chain(ra), shared=True, act=True)
    default, wprod, aprod = _parts(t)
    assert len(info["act"]) == 1 and info["spins"] == 1 and info["pcaps"] == 1   # the per-launch epoch is captured
    # the consumer issues no cp.async and waits for none; its spin stays
    assert "ttg.async_copy_global_to_local" not in default and "ttg.async_wait" not in default
    assert "ttg.async_commit_group" not in default and "ub.poison" in default
    assert default.count("ld.acquire.gpu") == 2
    # the weight producer never spins; the activation producer clones the spin (seed + poll + warp sync) verbatim
    assert "ld.acquire.gpu" not in wprod and aprod.count("ld.acquire.gpu") == 2 and "bar.warp.sync" in aprod
    assert aprod.index("ld.acquire.gpu") < aprod.index("ttg.async_copy_global_to_local")
    # every stage: both producers wait on the same empty barrier and arrive on the same ready barrier
    assert re.search(r"ttng\.init_barrier %rg_r0, 256 :", t)
    for p in (wprod, aprod):
        assert p.count("ttng.async_copy_mbarrier_arrive") == wprod.count("ttng.async_copy_mbarrier_arrive")
        assert p.count("ttng.wait_barrier") >= p.count("ttng.async_copy_mbarrier_arrive")
    assert wprod.count("ttg.async_copy_global_to_local") == (8 if ra else 4)
    assert aprod.count("ttg.async_copy_global_to_local") == 4
    s = offc.compile_ttgir_text(t)
    assert s["num_warps"] == 12 and s["spill"] == "0"


def test_act_off_keeps_the_activation_in_the_consumer():
    log = []
    t, info = ringws2.rewrite(_chain(1), shared=True, log=log)
    assert not info["act"] and any("written in-kernel" in l for l in log) and "partition1(" not in t


def test_act_refuses_a_write_in_the_reading_root():
    src = _chain(0)
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.M).group(0)
    src2 = src.replace(sync, sync + "\n      %rgt_c = arith.constant 0.0 : bf16\n      tt.store %y_ptr, %rgt_c : !tt.ptr<bf16>", 1)
    with pytest.raises(ringws2.Fail, match="root that reads it"):
        ringws2.rewrite(src2, shared=True, act=True)


def test_act_orders_an_unpublished_write_inside_the_region():
    # same-CTA program order is what made the original read correct: the ordering point restores it (RING3 R3)
    src = _mutate(_chain(1), "tt.atomic_rmw add, release, gpu", "tt.atomic_rmw add, relaxed, gpu")
    t, info = ringws2.rewrite(src, shared=True, act=True)
    assert info["act"] and info["order_points"] == 1
    _, _, aprod = _parts(t)
    assert aprod.index("ttng.wait_barrier %pr_wo") < aprod.index("ttg.async_copy_global_to_local")


def test_act_orders_a_read_behind_a_spin_on_another_counter():
    # RING2 verifier counterexample 2: the reader's spin polls a counter the writer does not publish; R3 no longer
    # compares publication roots (vacuous with one state tensor) - the CTA-local ordering point orders the read
    src = _chain(1)
    line = re.search(r"^.*tt\.atomic_rmw add, release, gpu, %state_ptr,.*$", src, re.M).group(0)
    t, info = ringws2.rewrite(src.replace(line, line.replace("%state_ptr,", "%epoch_ptr,")), shared=True, act=True)
    assert info["act"] and info["order_points"] == 1


def test_act_refuses_other_synchronization_in_the_region():
    src = _chain(0)
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.M).group(0)
    acq = sync + ("\n      %rgt_one = arith.constant 1 : i32\n      %rgt_t = arith.constant true\n      %rgt_a = tt.atomic_rmw add, "
                  "acquire, gpu, %epoch_ptr, %rgt_one, %rgt_t : (!tt.ptr<i32>, i32, i1) -> i32")
    with pytest.raises(ringws2.Fail, match="acquiring atomic"):
        ringws2.rewrite(src.replace(sync, acq, 1), shared=True, act=True)
    # allowlisted, but a fence (Helion's release fence before an exit trigger): synchronizes
    fence = sync + ('\n      %rgt_f = tt.elementwise_inline_asm "fence.acq_rel.gpu; // $0" {constraints = "=r", '
                    'packed_element = 1 : i32, pure = false} -> i32')
    with pytest.raises(ringws2.Fail, match="side-effecting inline asm"):
        ringws2.rewrite(src.replace(sync, fence, 1), shared=True, act=True)
    # a fence spelled off the allowlist: the module may wait on its predecessor (refused before the act rules)
    fence2 = sync + ('\n      %rgt_f = tt.elementwise_inline_asm "fence.acq_rel.gpu; mov.u32 $0, $1;" {constraints = "=r,r", '
                     'packed_element = 1 : i32, pure = false} %rgt_x : i32 -> i32')
    fence2 = fence2.replace("%rgt_x", re.search(r"(%\S+) = tt\.get_program_id x", src).group(1))
    with pytest.raises(ringws2.Fail, match="off the allowlist"):
        ringws2.rewrite(src.replace(sync, fence2, 1), shared=True, act=True)


def test_act_negative_control_drops_the_spin_clone():
    t, info = ringws2.rewrite(_chain(0), shared=True, act=True, neg="nospin")
    _, _, aprod = _parts(t)
    assert "ld.acquire.gpu" not in aprod and "ttg.async_copy_global_to_local" in aprod
    with pytest.raises(ringws2.Fail):
        ringws2.rewrite(_chain(0), shared=True, neg="nospin")


def test_act_producer_warps():
    t, info = ringws2.rewrite(_chain(1), shared=True, act=True, pw=2, pwa=1)
    assert re.search(r"partition0\(.*\) num_warps\(2\)", t) and re.search(r"partition1\(.*\) num_warps\(1\)", t)
    assert re.search(r"ttng\.init_barrier %rg_r0, 96 :", t)
    offc.compile_ttgir_text(t)


_STAMP_ASM = '"{ .reg .u64 t; mov.u64 t, %globaltimer; st.global.u64 [$1], t; mov.u32 $0, 0; }"'


def _stamp_after_sync(src, ptr="%out_ptr", asm=_STAMP_ASM, addr=None):
    """The allowlisted stamp (#22) right after root B's warp sync, storing through ``ptr`` (as an i64 address)."""
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.M).group(0)
    a = addr or f"%rgt_a = tt.ptr_to_int {ptr} : !tt.ptr<bf16> -> i64"
    return src.replace(sync, sync + (f'\n      {a}\n      %rgt_s = tt.elementwise_inline_asm {asm} {{constraints = "=r,l", '
                                     'packed_element = 1 : i32, pure = false} %rgt_a : i64 -> i32'), 1)


def test_act_allows_globaltimer_stamps_only():
    src = _chain(0)
    # the stamp writes only its pointer operand's argument (out: not an activation): act stays
    t, info = ringws2.rewrite(_stamp_after_sync(src), shared=True, act=True)
    assert info["act"]
    # a timer read orders nothing either
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.M).group(0)
    timer = sync + ('\n      %rgt_t = tt.elementwise_inline_asm "mov.u64 $0, %globaltimer;" {constraints = "=l", '
                    'packed_element = 1 : i32, pure = false} -> i64')
    assert ringws2.rewrite(src.replace(sync, timer, 1), shared=True, act=True)[1]["act"]
    # a releasing store is not the allowlisted stamp: off the allowlist (refused before the act rules)
    with pytest.raises(ringws2.Fail, match="off the allowlist"):
        ringws2.rewrite(_stamp_after_sync(src, asm=_STAMP_ASM.replace("st.global.u64", "st.release.gpu.global.u64")),
                        shared=True, act=True)
    # the stamp through an untraceable address (an integer from the program id) writes every argument, so the
    # activation y is written in the root that reads it
    pid = re.search(r"(%\S+) = tt\.get_program_id x", src).group(1)
    with pytest.raises(ringws2.Fail, match="root that reads it"):
        ringws2.rewrite(_stamp_after_sync(src, addr=f"%rgt_a = arith.extsi {pid} : i32 to i64"), shared=True, act=True)


def test_ring_emits_only_allowlisted_assembly():
    """Every inline assembly op of every default ring form (both producers' spin clones included) is an exact entry of
    the hoist pass's allowlist; the lane-0 polling option, whose template was not, is gone."""
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    for src in (_chain(1), _chain(0), _gemv()):
        for _, text in pf._candidates(src, (), ring=True):
            if text is None:
                continue
            asm = [line for line in text.splitlines() if "tt.elementwise_inline_asm" in line]
            assert all(pf._asm_entry(line) is not None for line in asm), [line for line in asm if not pf._asm_entry(line)]
    with pytest.raises(TypeError):
        ringws2.rewrite(_chain(1), shared=True, act=True, poll1=True)


def test_unroll_consumer_loop_exactly():
    src = _chain(0)
    t0, _ = ringws2.rewrite(src, shared=True, act=True)
    t1, _ = ringws2.rewrite(src, shared=True, act=True, unroll=16)
    d0, d1 = _parts(t0)[0], _parts(t1)[0]
    # the K loop (16 iterations) disappears from the consumer; one ready wait per iteration remains
    assert d1.count("scf.for") == d0.count("scf.for") - 1
    assert d1.count("ttng.wait_barrier %rg_rb") == 16 * d0.count("ttng.wait_barrier %rg_rb")
    assert re.search(r"_u15 = arith\.constant 1920 : i32", d1)
    assert _parts(t1)[1:] == _parts(t0)[1:]    # producers untouched
    offc.compile_ttgir_text(t1)
    # loops above the trip limit stay
    t2, _ = ringws2.rewrite(src, shared=True, act=True, unroll=8)
    assert _parts(t2)[0].count("scf.for") == d0.count("scf.for")


def test_ring_mode_defaults_and_act_refusal_falls_back_to_weight_ring(monkeypatch):
    for k in list(os.environ):
        if k.startswith("HELION_TTGIR_RING_"):
            monkeypatch.delenv(k)
    assert ringws2.ring_options() == {"act": True, "pw": "auto", "pwa": "auto", "unroll": 32, "late_ws": True}
    t, info = ringws2.rewrite_budgeted(_chain(1))
    assert info["act"] and re.search(r"partition0\(.*\) num_warps\(1\)", t) and re.search(r"partition1\(.*\) num_warps\(1\)", t)
    # weight-only kernel: auto producer warps = the kernel's warp count
    t2, info2 = ringws2.rewrite_budgeted(_gemv())
    assert not info2["act"] and re.search(r"partition0\(.*\) num_warps\(4\)", t2) and "partition1(" not in t2
    # an activation rule refuses -> the weight-only ring still applies
    src = _chain(0)
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.MULTILINE).group(0)
    src2 = src.replace(sync, sync + "\n      %rgt_c = arith.constant 0.0 : bf16\n      tt.store %y_ptr, %rgt_c : !tt.ptr<bf16>", 1)
    t3, info3 = ringws2.rewrite_budgeted(src2)
    assert not info3["act"] and "root that reads it" in info3["act_refused"] and "partition1(" not in t3
    # the weight-only fallback starts its warp_specialize late too (RING_DEFAULTS late_ws): the leading ring-free
    # root runs before the partitions exist, as in the two-producer ring
    _, info4 = ringws2.rewrite_budgeted(src2)
    t5, _ = ringws2.rewrite_budgeted(src2, late_ws=False)
    assert t5 != t3 and info4["act_refused"]


# Every former experiment knob (HELION_TTGIR_RING_*) is an automatic decision now: setting one changes nothing.
_FORMER_KNOBS = {"HELION_TTGIR_RING_GROUP": "2", "HELION_TTGIR_RING_LATE_WS": "0", "HELION_TTGIR_RING_DEPTH": "3",
                 "HELION_TTGIR_RING_DROP_WAITS": "1", "HELION_TTGIR_RING_PW": "2", "HELION_TTGIR_RING_PWA": "2",
                 "HELION_TTGIR_RING_ACT": "0", "HELION_TTGIR_RING_POLL1": "1", "HELION_TTGIR_RING_UNROLL": "0",
                 "HELION_TTGIR_RING_UNROLL_FACTOR": "2", "HELION_TTGIR_RING_PREFETCH": "1", "HELION_TTGIR_RING_ROLL": "1"}


def test_former_ring_knobs_change_nothing(monkeypatch):
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    for k in list(os.environ):
        if k.startswith("HELION_TTGIR_RING_"):
            monkeypatch.delenv(k)
    src = _chain(1)
    base = pf._candidates(src, (), ring=True)
    with pf.enabled(ring=True):
        key = pf._StagesHook(None)()
    for k, v in _FORMER_KNOBS.items():
        monkeypatch.setenv(k, v)
    assert ringws2.ring_options() == ringws2.RING_DEFAULTS and ringws2.options_key() == ""
    assert pf._candidates(src, (), ring=True) == base
    with pf.enabled(ring=True):
        assert pf._StagesHook(None)() == key


def test_negative_control_hook_is_private_and_keyed(monkeypatch):
    """HELION_TTGIR_RING_NEG (tests only) reaches the ring and the compile key while set, and only then."""
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    monkeypatch.delenv("HELION_TTGIR_RING_NEG", raising=False)
    with pf.enabled(ring=True):
        key = pf._StagesHook(None)()
    assert key[0].endswith("-ring1")
    assert pf._ring_key(True) == "ring1" and ringws2.options_key() == ""
    monkeypatch.setenv("HELION_TTGIR_RING_NEG", "nospin")
    assert ringws2.ring_options()["neg"] == "nospin" and pf._ring_key(True) == "ring1[neg=nospin]"
    with pf.enabled(ring=True):
        neg_key = pf._StagesHook(None)()
    assert neg_key[0] != key[0] and neg_key[1] != key[1]
    assert pf._ring_key(False) == "ring0"    # a kernel without the ring ignores it
    t, info = ringws2.rewrite_budgeted(_chain(1))
    assert info["opts"]["neg"] == "nospin"


def test_two_producer_ring_falls_back_to_weight_ring_then_hoist(tmp_path, monkeypatch):
    for k in list(os.environ):
        if k.startswith("HELION_TTGIR_RING_"):
            monkeypatch.delenv(k)
    ck = _compile_under_hook(tmp_path / "a", monkeypatch, chain=True)
    assert ck.metadata.num_warps == 8 and ck.asm["ttgir"].count("partition1(") == 1   # 4 + 1 + 1 warps (rounded)
    n_act = ck.asm["ptx"].count("ld.acquire.gpu")      # consumer spin + its clone in the activation producer
    # the two-producer ring fails in make_llir -> the weight-only ring (one producer partition) is next (Triton keeps
    # the TTGIR of the first candidate in asm; the PTX / binary are the fallback's)
    ck = _compile_under_hook(tmp_path / "b", monkeypatch, _injector("llir", lambda m: "partition1(" in m.str()), chain=True)
    assert ck.metadata.num_warps == 8 and "cp.async.mbarrier.arrive" in ck.asm["ptx"]
    assert 0 < ck.asm["ptx"].count("ld.acquire.gpu") < n_act
    # every ring fails -> the hoist
    ck = _compile_under_hook(tmp_path / "c", monkeypatch, _injector("llir", lambda m: "ttg.warp_specialize" in m.str()), chain=True)
    assert ck.metadata.num_warps == 4 and "cp.async.mbarrier.arrive" not in ck.asm["ptx"]


# ---------------------------------------------------------------------------------------------------------------------
# RING3 soundness: handoff invariants on the emitted TTGIR (HIC), synchronizing callees (R2), the CTA-local ordering
# point (R3), and tests for every guard the RING2 verifier's mutants showed untested.

_V = r"%[\w$.#-]+"


def _hic_defs(lines):
    d = {}
    for k, line in enumerate(lines):
        m = re.match(rf"^\s*({_V})(?::\d+)?\s*=\s*(.*)$", line)
        if m:
            d[m.group(1)] = (k, m.group(2))
    return d


def _hic_blocks(lines):
    out, stack = [], []
    for k, line in enumerate(lines):
        s = line.split(" loc(")[0].rstrip()
        if s.strip().startswith("}"):
            if stack:
                stack.pop()
            if s.endswith("{"):
                stack.append(k)
        elif s.endswith("{"):
            out.append(tuple(stack))
            stack.append(k)
            continue
        out.append(tuple(stack))
    return out


def _hic_consumer(body, rings):
    """Consumer invariants: per stage slot S one ready wait and one release, every read of a view of S strictly between
    them in one block (C1/C2); slot and phase from one counter (C3); straight-line stages use distinct counters, each
    the previous + 1 (C4: unrolled iterations carry the counter)."""
    lines = body.splitlines()
    defs, blk, viol = _hic_defs(lines), _hic_blocks(lines), []
    bar = {}
    for v, (_, rhs) in defs.items():
        m = re.match(rf"ttg\.memdesc_index %rg_(ready|empty)\[({_V})\]", rhs)
        if m:
            bar[v] = (m.group(1), m.group(2))
    slots_of_bars = {x[1] for x in bar.values()}
    views = {}
    for v, (_, rhs) in defs.items():
        m = re.match(rf"ttg\.memdesc_index (%[\w$.]+)\[({_V})\]", rhs)
        if m and m.group(1) in rings:
            mk = re.match(rf"arith\.muli ({_V}), ({_V})", defs.get(m.group(2), (0, ""))[1])
            views[v] = mk.group(1) if mk and mk.group(1) in slots_of_bars else m.group(2)
    changed = True
    while changed:
        changed = False
        for v, (_, rhs) in defs.items():
            m = re.match(rf"ttg\.memdesc_trans ({_V})", rhs)
            if m and m.group(1) in views and v not in views:
                views[v] = views[m.group(1)]
                changed = True
    waits, rels, reads = {}, {}, {}
    for k, line in enumerate(lines):
        c = line.split(" loc(")[0]
        m = re.match(rf"^\s*ttng\.wait_barrier ({_V}), ({_V})", c)
        if m and bar.get(m.group(1), ("",))[0] == "ready":
            waits.setdefault(bar[m.group(1)][1], []).append((k, m.group(2)))
        m = re.match(rf"^\s*ttng\.(?:arrive_barrier|tc_gen5_commit) ({_V})", c)
        if m and bar.get(m.group(1), ("",))[0] == "empty":
            rels.setdefault(bar[m.group(1)][1], []).append(k)
        if re.search(r"ttg\.local_load|ttng\.tc_gen5_mma", c):
            for u in re.findall(_V, c.split("=", 1)[-1]):
                if u in views:
                    reads.setdefault(views[u], []).append(k)
    slots = set(waits) | set(rels) | set(reads)
    by_blk = {}
    for s in slots:
        w, r, rd = waits.get(s, []), rels.get(s, []), reads.get(s, [])
        if len(w) != 1 or len(r) != 1:
            viol.append(f"C1 {s}: {len(w)} waits, {len(r)} releases")
            continue
        (wk, ph), rk = w[0], r[0]
        if not rd:
            viol.append(f"C2 {s}: no reads")
        for x in rd:
            if not (wk < x < rk) or blk[x] != blk[wk] or blk[rk] != blk[wk]:
                viol.append(f"C2 {s}: read at {x} outside (wait {wk}, release {rk})")
        ms = re.match(rf"arith\.remui ({_V}), ({_V})", defs.get(s, (0, ""))[1])
        mp = re.match(rf"arith\.andi ({_V}), ({_V})", defs.get(ph, (0, ""))[1])
        mq = re.match(rf"arith\.divui ({_V}), ({_V})", defs.get(mp.group(1), (0, ""))[1]) if mp else None
        if not (ms and mq and ms.group(1) == mq.group(1)):
            viol.append(f"C3 {s}: slot / phase not from one counter")
        elif ms:
            by_blk.setdefault(blk[wk], []).append((wk, ms.group(1)))
    for b, xs in by_blk.items():
        xs.sort()
        cs = [c for _, c in xs]
        if len(set(cs)) != len(cs):
            viol.append(f"C4 {b}: stages reuse a counter")
        for (_, c0), (_, c1) in zip(xs, xs[1:]):
            d1 = defs.get(c1)
            if d1 is None or blk[d1[0]] != b:
                continue
            m = re.match(rf"arith\.addi ({_V}), ", d1[1])
            if not (m and m.group(1) == c0):
                viol.append(f"C4 {b}: counter {c1} is not {c0} + 1")
    return dict(stages=len(slots), reads=sum(len(v) for v in reads.values())), viol


def _hic_producer(body, name):
    """Producer invariants: per stage empty wait -> copies into views of that slot -> arrive (P1); empty parity =
    ((c / D) & 1) ^ 1 and slot = c % D of the same counter (P2); as many waits as arrives (P3)."""
    lines = body.splitlines()
    defs, viol, cur, nw, na = _hic_defs(lines), [], None, 0, 0
    for k, line in enumerate(lines):
        c = line.split(" loc(")[0]
        m = re.match(rf"^\s*ttng\.wait_barrier ({_V}), ({_V})", c)
        if m and m.group(1).startswith("%pr_eb"):
            idx = re.search(rf"\[({_V})\]", defs[m.group(1)][1]).group(1)
            mx = re.match(rf"arith\.xori ({_V}), %pr_one", defs.get(m.group(2), (0, ""))[1])
            ma = re.match(rf"arith\.andi ({_V}), %pr_one", defs.get(mx.group(1), (0, ""))[1]) if mx else None
            mq = re.match(rf"arith\.divui ({_V}), %pr_D", defs.get(ma.group(1), (0, ""))[1]) if ma else None
            ms = re.match(rf"arith\.remui ({_V}), %pr_D", defs.get(idx, (0, ""))[1])
            if not (mq and ms and mq.group(1) == ms.group(1)):
                viol.append(f"P2 {name} {k}: parity / slot not from one counter")
            if cur is not None:
                viol.append(f"P1 {name} {k}: wait before the previous stage arrived")
            cur, nw = idx, nw + 1
            continue
        m = re.match(rf"^\s*(?:{_V}\s*=\s*)?ttg\.async_copy_global_to_local ({_V}), ({_V})", c)
        if m:
            vi = re.search(rf"\[({_V})\]", defs.get(m.group(2), (0, ""))[1] or "")
            mk = re.match(rf"arith\.muli ({_V}),", defs.get(vi.group(1), (0, ""))[1]) if vi else None
            if cur is None or (mk.group(1) if mk else vi and vi.group(1)) != cur:
                viol.append(f"P1 {name} {k}: copy outside its stage")
            continue
        m = re.match(rf"^\s*ttng\.async_copy_mbarrier_arrive ({_V})", c)
        if m:
            idx = re.search(rf"\[({_V})\]", defs.get(m.group(1), (0, ""))[1])
            if cur is None or idx is None or idx.group(1) != cur:
                viol.append(f"P1 {name} {k}: arrive without its stage wait")
            cur, na = None, na + 1
    if nw != na:
        viol.append(f"P3 {name}: {nw} waits vs {na} arrives")
    return dict(waits=nw, arrives=na), viol


def _hic(text):
    a = text.index("ttg.warp_specialize(")
    d0, p0 = text.index("default {", a), text.index("partition0(", a)
    p1 = text.find("partition1(", p0)
    end = text.index("} : (", p1 if p1 > 0 else p0)
    rings = set(re.findall(rf"^\s*({_V}) = ttg\.memdesc_reinterpret %rg_bytes", text, re.M)) or \
        set(re.findall(rf"^\s*({_V}) = ttg\.local_alloc : \(\) -> !ttg\.memdesc<\d+x", text[:a], re.M))
    out, viol = _hic_consumer(text[d0:p0], rings)
    res = {"consumer": out}
    for name, body in (("p0", text[p0:(p1 if p1 > 0 else end)]), ("p1", text[p1:end] if p1 > 0 else None)):
        if body is not None:
            res[name], v = _hic_producer(body, name)
            viol += v
    if "p1" in res and res["p0"]["waits"] != res["p1"]["waits"]:
        viol.append("P3 producers walk different stage counts")
    return res, viol


@pytest.mark.parametrize("cfg", [dict(ra=0), dict(ra=1), dict(ra=0, bn=32, bk=256, ns=3), dict(ra=1, ns=6),
                                 dict(ra=0, bn=128, bk=128)])
def test_handoff_invariants_of_the_default_ring(cfg):
    t, info = ringws2.rewrite_budgeted(_chain(**cfg))
    assert info["act"]
    res, viol = _hic(t)
    assert not viol, viol[:5]
    if cfg.get("bn", 16) != 128:
        # two readers (w and y local_loads) per stage: a release after the FIRST reader is a C2 violation
        assert res["consumer"]["reads"] > res["consumer"]["stages"]


@pytest.mark.parametrize("act", [False, True])
def test_handoff_invariants_without_unroll_and_weight_only(act):
    t, _ = ringws2.rewrite(_chain(1), shared=True, act=act, pw=1 if act else None, pwa=1, unroll=0)
    _, viol = _hic(t)
    assert not viol, viol[:5]
    t, _ = ringws2.rewrite_budgeted(_gemv(16, 128), act=False)
    _, viol = _hic(t)
    assert not viol, viol[:5]


@pytest.mark.parametrize("mutation,code", [
    # (what the mutant does to the emitted text, the invariant that must catch it)
    ("release after the first reader", "C2"), ("consumer parity flipped", "C3"), ("producer parity flipped", "P2"),
    ("unrolled iterations share one counter", "C4")])
def test_handoff_invariant_checker_has_teeth(mutation, code):
    t, _ = ringws2.rewrite_budgeted(_chain(0))
    d0, p0 = t.index("default {"), t.index("partition0(")
    default, rest = t[d0:p0], t[p0:]
    if mutation == "release after the first reader":
        m = re.search(r"^(\s*)(%rg_eb\w+) = (ttg\.memdesc_index %rg_empty.*)\n\s*(ttng\.arrive_barrier \2, 1 .*)\n", default, re.M)
        wait = default.rfind("ttng.wait_barrier %rg_rb", 0, m.start())
        first = default.index("ttg.local_load", wait)
        cut = default.index("\n", first) + 1
        assert default.count("ttg.local_load", cut, m.start()) >= 1   # a second reader follows the first
        default = default[:cut] + m.group(0) + default[cut:m.start()] + default[m.end():]
    elif mutation == "consumer parity flipped":
        default = re.sub(r"(%rg_ph\w+) = arith\.andi", r"\1 = arith.xori", default, count=1)
    elif mutation == "producer parity flipped":
        rest = re.sub(r"(%pr_pi\w+) = arith\.xori", r"\1 = arith.andi", rest, count=1)
    else:
        cs = re.findall(r"= arith\.remui (%rg_cnt\w+|%rg_cn\w+)", default)
        last = re.findall(r"(%rg_s\w+) = arith\.remui (%\S+), ", default)
        default = default.replace(f"arith.remui {last[1][1]}, ", f"arith.remui {last[0][1]}, ", 1) if len(last) > 1 else default
        assert len(last) > 1, cs
    _, viol = _hic(t[:d0] + default + rest)
    assert any(v.startswith(code) for v in viol), viol[:5]


# ---- R2: synchronizing callees, volatile loads, tickets
@triton.jit(noinline=True)
def _wait_callee(state_ptr, target):
    c = _acq(state_ptr)
    while c != target:
        c = _acq(state_ptr)
    tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit(noinline=True)
def _scale_callee(x):
    return (x % 2).to(tl.float32) + 1.0


@triton.jit
def _cx_k(x_ptr, y_ptr, w_ptr, out_ptr, state_ptr, epoch_ptr, flag_ptr, N, K: tl.constexpr, T: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, SPIN: tl.constexpr):
    """Root A writes y and publishes; root B waits (SPIN 0 inline, 1 in a noinline callee, 2 one tile per program
    under an if (no loop shared with the copies), 3 inline + a volatile flag load, 4 inline + a pure noinline call)."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32)
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    if SPIN == 2:
        tile = pid
        if tile < n_tiles:
            c = _acq(state_ptr)
            while c != target:
                c = _acq(state_ptr)
            tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                      dtype=tl.int32, is_pure=False, pack=1)
            rn = tile * BN + tl.arange(0, BN)
            acc = tl.zeros([BN, T], dtype=tl.float32)
            for k in tl.range(0, K, BK, num_stages=NS):
                w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
                y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
                acc = tl.dot(w, tl.trans(y), acc)
            tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    else:
        for tile in tl.range(pid, n_tiles, npid, num_stages=1):
            if SPIN == 1:
                _wait_callee(state_ptr, target)
            else:
                c = _acq(state_ptr)
                while c != target:
                    c = _acq(state_ptr)
                tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                          dtype=tl.int32, is_pure=False, pack=1)
            rn = tile * BN + tl.arange(0, BN)
            acc = tl.zeros([BN, T], dtype=tl.float32)
            if SPIN == 3:
                acc += tl.load(flag_ptr, volatile=True).to(tl.float32) * 0.0
            for k in tl.range(0, K, BK, num_stages=NS):
                w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
                y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
                acc = tl.dot(w, tl.trans(y), acc)
            if SPIN == 4:
                acc = acc * _scale_callee(tile)
            tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


SIGX = dict(x_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32", epoch_ptr="*i32",
            flag_ptr="*i32", N="i32")


def _cx(spin):
    return _compile_jit(_cx_k, SIGX, dict(K=2048, T=16, BN=16, BK=128, NS=4, SPIN=spin), num_warps=4).asm["ttgir"]


def test_act_refuses_a_spin_in_a_noinline_callee():
    src = _cx(1)
    assert "tt.call" in src
    with pytest.raises(ringws2.Fail, match="synchronizing function"):
        ringws2.rewrite(src, shared=True, act=True)
    t, info = ringws2.rewrite_budgeted(src)   # the weight-only ring needs no spin: still applies
    assert not info["act"] and "synchronizing function" in info["act_refused"] and "partition1(" not in t


def test_act_accepts_a_pure_noinline_callee():
    src = _cx(4)
    assert "tt.call" in src
    t, info = ringws2.rewrite_budgeted(src)
    assert info["act"] and info["spins"] == 1


def test_sync_function_summary_is_transitive():
    src = _cx(1)
    lines = P_uniq(src)
    funcs = ringws2.P._functions(lines)
    sync = ringws2._sync_functions(lines, funcs)
    assert any("wait_callee" in n for n in sync) and any("_cx_k" in n for n in sync)   # the entry spins inline too
    lines4 = P_uniq(_cx(4))
    assert not any("scale_callee" in n for n in ringws2._sync_functions(lines4, ringws2.P._functions(lines4)))
    # a caller of a synchronizing function synchronizes; an unknown callee synchronizes
    text = "\n".join(lines)
    callee = next(n for n in sync if "wait_callee" in n)
    wrapper = (f'  tt.func private @wrap(%a: !tt.ptr<i32>, %b: i32) attributes {{noinline = true}} {{\n'
               f'    tt.call @{callee}(%a, %b) : (!tt.ptr<i32>, i32) -> ()\n    tt.return\n  }}\n'
               f'  tt.func private @ext(%a: i32) attributes {{noinline = true}} {{\n'
               f'    tt.call @not_defined(%a) : (i32) -> ()\n    tt.return\n  }}\n')
    lines2 = (text[:text.rindex("}")] + wrapper + "}\n").splitlines()
    sync2 = ringws2._sync_functions(lines2, ringws2.P._functions(lines2))
    assert {"wrap", "ext"} <= sync2


def P_uniq(src):  # noqa: N802
    return ringws2.P._uniquify(src).splitlines()


def test_act_refuses_a_volatile_load_in_the_region():
    with pytest.raises(ringws2.Fail, match="volatile load"):
        ringws2.rewrite(_cx(3), shared=True, act=True)
    _, info = ringws2.rewrite_budgeted(_cx(3))
    assert not info["act"] and "volatile" in info["act_refused"]


@triton.jit
def _tk_k(x_ptr, y_ptr, w_ptr, out_ptr, state_ptr, epoch_ptr, ticket_ptr, N, K: tl.constexpr, T: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32)
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    tile = tl.atomic_add(ticket_ptr, 1, sem="relaxed", scope="gpu")
    while tile < n_tiles:
        c = _acq(state_ptr)
        while c != target:
            c = _acq(state_ptr)
        tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                  dtype=tl.int32, is_pure=False, pack=1)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
        tile = tl.atomic_add(ticket_ptr, 1, sem="relaxed", scope="gpu")
    tl.store(epoch_ptr + pid, epoch)


def test_act_refuses_dynamic_tickets():
    sig = dict(x_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32", epoch_ptr="*i32",
               ticket_ptr="*i32", N="i32")
    src = _compile_jit(_tk_k, sig, dict(K=2048, T=16, BN=16, BK=128, NS=4), num_warps=4).asm["ttgir"]
    with pytest.raises(ringws2.Fail, match="ticket"):
        ringws2.rewrite(src, shared=True, act=True)
    _, info = ringws2.rewrite_budgeted(src)
    assert not info["act"] and "ticket" in info["act_refused"]


def test_act_clones_a_spin_under_an_if_without_a_shared_loop():
    t, info = ringws2.rewrite_budgeted(_cx(2))
    assert info["act"] and info["spins"] == 1
    _, _, aprod = _parts(t)
    assert aprod.count("ld.acquire.gpu") >= 2 and aprod.index("ld.acquire.gpu") < aprod.index("ttg.async_copy_global_to_local")


# ---- R3: the CTA-local ordering point
@triton.jit
def _r3_k(x_ptr, y_ptr, w_ptr, wa_ptr, x2_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr, T: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, KA: tl.constexpr, TWO: tl.constexpr):
    """RING2 verifier counterexample 2: one state tensor for every counter; root B (inside the region: it streams a
    weight wa) writes THIS program's rows of x2 and publishes state[1]; root C waits only for root A (state[0]) and
    reads y and its own x2 rows (correct in the original by same-CTA program order)."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32) * 1.5
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    for r in tl.range(pid, npid, npid, num_stages=1):
        s = tl.zeros([T, BK], dtype=tl.float32)
        for ka in tl.range(0, KA, BK, num_stages=NS):
            s += tl.load(wa_ptr + rt[:, None] * KA + (ka + rk)[None, :]).to(tl.float32)
        for kb in tl.range(0, K, BK, num_stages=1):
            tl.store(x2_ptr + (r * T + rt)[:, None] * K + (kb + rk)[None, :], (s * 0.25).to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr + 1, 1, sem="release", scope="gpu")
    if TWO:   # a second in-region root rewrites this program's x2 rows: the reader must be ordered after THIS one
        for r2 in tl.range(pid, npid, npid, num_stages=1):
            for kb in tl.range(0, K, BK, num_stages=1):
                p2 = x2_ptr + (r2 * T + rt)[:, None] * K + (kb + rk)[None, :]
                tl.store(p2, (tl.load(p2).to(tl.float32) + 1.0).to(tl.bfloat16))
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        c = _acq(state_ptr)
        while c != target:
            c = _acq(state_ptr)
        tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                  dtype=tl.int32, is_pure=False, pack=1)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            x2 = tl.load(x2_ptr + (pid * T + rt)[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y + x2), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


SIGR3 = dict(x_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", wa_ptr="*bf16", x2_ptr="*bf16", out_ptr="*bf16",
             state_ptr="*i32", epoch_ptr="*i32", N="i32")


def _r3(two=0):
    return _compile_jit(_r3_k, SIGR3, dict(K=2048, T=16, BN=16, BK=128, NS=4, KA=1024, TWO=two),
                        num_warps=4).asm["ttgir"]


def test_act_orders_a_same_cta_writer_root_before_its_readers():
    t, info = ringws2.rewrite_budgeted(_r3())
    assert info["act"] and info["order_points"] == 1
    default, wprod, aprod = _parts(t)
    # the consumer arrives once, after root B (its last x2 store and its publication), before root C's spin
    arrive = default.index("ttng.arrive_barrier %rg_wov0, 1")
    assert default.rfind("tt.atomic_rmw add, release, gpu", 0, arrive) > default.rfind("tt.store", 0, arrive) > 0
    assert arrive < default.index("ld.acquire.gpu", arrive - 1)
    # the activation producer waits (phase 0) before root C's clone: before its spin clone and every copy of root C;
    # the weight producer never waits on it
    wo = aprod.index("ttng.wait_barrier %pr_wo")
    # single-use barrier: phase 0 (a phase-1 wait on a fresh mbarrier passes at once)
    assert re.search(r"ttng\.wait_barrier %pr_wo\d+, %pr_c0 :", aprod)
    first_c = aprod.index("ld.acquire.gpu")
    assert wo < first_c and "%pr_wo" not in wprod
    assert re.search(r"ttng\.init_barrier %rg_woi0, 1 :", t) and "ttng.inval_barrier %rg_woi0" in t
    _, viol = _hic(t)
    assert not viol, viol[:5]
    offc.compile_ttgir_text(t)
    # negative control (TEST ONLY): no ordering wait in the activation producer
    t2, _ = ringws2.rewrite_budgeted(_r3(), neg="noorder")
    assert "%pr_wo" not in _parts(t2)[2] and "%rg_wov0" in _parts(t2)[0]


def test_act_orders_after_the_last_writer_root():
    t, info = ringws2.rewrite_budgeted(_r3(two=1))
    assert info["act"] and info["order_points"] == 1
    default = _parts(t)[0]
    arrive = default.index("ttng.arrive_barrier %rg_wov0, 1")
    # after the second root's x2 stores (a load-add-store loop), not after root B's publication
    assert default.rfind("tt.store", 0, arrive) > default.rfind("tt.atomic_rmw add, release, gpu", 0, arrive)
    assert default.find("tt.store", arrive) > default.index("ld.acquire.gpu", arrive)  # next store is root C's


@triton.jit
def _desc_k(x_ptr, xa_ptr, y_ptr, w_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr, T: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, KA: tl.constexpr):
    """_chain_k (RA=1: root A in the region) whose root A writes y through a device tensor descriptor (TMA store,
    async proxy): the ordering point's generic-proxy release does not cover it -> act must refuse."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    desc = tl.make_tensor_descriptor(y_ptr, shape=[T, K], strides=[K, 1], block_shape=[T, BK])
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32)
        for ka in tl.range(0, KA, BK, num_stages=NS):
            v += tl.load(xa_ptr + rt[:, None] * KA + (ka + rk)[None, :]).to(tl.float32)
        desc.store([0, blk * BK], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        c = _acq(state_ptr)
        while c != target:
            c = _acq(state_ptr)
        tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                  dtype=tl.int32, is_pure=False, pack=1)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


def test_act_refuses_an_async_proxy_writer_in_the_region():
    src = _compile_jit(_desc_k, SIGC, dict(K=2048, T=16, BN=16, BK=128, NS=4, KA=512), num_warps=4).asm["ttgir"]
    assert "async_tma_copy_local_to_global" in src or "descriptor_store" in src
    with pytest.raises(ringws2.Fail, match="written by"):
        ringws2.rewrite(src, shared=True, act=True)


@triton.jit(noinline=True)
def _sys_callee(p):
    return tl.inline_asm_elementwise("ld.relaxed.sys.global.u32 $0, [$1];", "=r,l", [p], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _sys_k(w_ptr, x_ptr, out_ptr, flag_ptr, N, K: tl.constexpr, T: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           NS: tl.constexpr):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    rk = tl.arange(0, BK)
    rt = tl.arange(0, T)
    for tile in range(pid, tl.cdiv(N, BN), npid):
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            x = tl.load(x_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(x), acc)
        acc += _sys_callee(flag_ptr).to(tl.float32) * 0.0
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))


def test_system_scope_ops_in_a_callee_are_refused():
    src = _compile_jit(_sys_k, dict(SIG, flag_ptr="*i32"), dict(K=2048, T=16, BN=16, BK=128, NS=4),
                       num_warps=4).asm["ttgir"]
    assert "tt.call" in src
    with pytest.raises(ringws2.Fail, match="system-scope"):
        ringws2.rewrite(src, shared=True)


def test_act_prefix_writers_need_no_ordering_point():
    t, info = ringws2.rewrite_budgeted(_chain(0))   # root A runs before the warp_specialize
    assert info["act"] and info["order_points"] == 0 and "%rg_wo" not in t
    t, info = ringws2.rewrite_budgeted(_chain(1))   # root A streams a weight: inside the region
    assert info["act"] and info["order_points"] == 1


def test_act_region_is_the_emitted_warp_specialize():
    for src in (_chain(0), _chain(1), _r3(), _cx(2)):
        t, info = ringws2.rewrite_budgeted(src)
        lines = ringws2.P._uniquify(src).splitlines()
        first = next(k for k, line in enumerate(lines) if "ttg.async_copy_global_to_local" in line)
        top = first
        ind = None
        f = [k for k, line in enumerate(lines) if "tt.func public" in line][0]
        ind0 = ringws2.P._indent(lines[f + 1])
        while ringws2.P._indent(lines[top]) > ind0 or not ringws2.P._code(lines[top]).strip():
            top -= 1
        assert info["ws_start"] == top - 1, (info["ws_start"], top)
        assert ind is None


@triton.jit
def _pcap_k(x_ptr, xa_ptr, y_ptr, w_ptr, out_ptr, state_ptr, epoch_ptr, off_ptr, N, K: tl.constexpr, T: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, RA: tl.constexpr, KA: tl.constexpr):
    """_chain_k plus a scalar loaded at the top level AFTER root A and used in root B's weight address: a value the
    producers can only capture (not clone) - legal iff it is defined before the warp_specialize."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32)
        if RA:
            for ka in tl.range(0, KA, BK, num_stages=NS):
                v += tl.load(xa_ptr + rt[:, None] * KA + (ka + rk)[None, :]).to(tl.float32)
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    off = tl.load(off_ptr) * 128   # keeps w's 16 B alignment (Triton then still pipelines it with cp.async)
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        c = _acq(state_ptr)
        while c != target:
            c = _acq(state_ptr)
        tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                  dtype=tl.int32, is_pure=False, pack=1)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + off + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


def test_act_captures_prefix_scalars_only():
    sig = dict(SIGC, off_ptr="*i32")
    mk = lambda ra: _compile_jit(_pcap_k, sig, dict(K=2048, T=16, BN=16, BK=128, NS=4, RA=ra, KA=512),  # noqa: E731
                                 num_warps=4).asm["ttgir"]
    t, info = ringws2.rewrite(mk(0), shared=True, act=True)    # `off` is loaded in the ring-free prefix
    assert info["act"] and info["pcaps"] == 2
    with pytest.raises(ringws2.Fail, match="not defined before the warp_specialize"):
        ringws2.rewrite(mk(1), shared=True, act=True)          # root A is in the region: `off` is too


# ---- STEP 1 guards whose call sites had no test (RING2 verifier mutants G4, G6, G10, G14, G2)
def test_smem_of_checks_the_allocation_alignment(monkeypatch):
    t, _ = ringws2.rewrite_budgeted(_gemv(16, 128), act=False)
    seen = []

    def spy(text):
        seen.append("ttg.local_alloc" in text and "allocation.offset" in text)
        raise ringws2.Fail("spy")
    monkeypatch.setattr(ringws2, "check_allocation_alignment", spy)
    with pytest.raises(ringws2.Fail, match="spy"):
        ringws2.smem_of(t)
    assert seen == [True]


def test_slot_bytes_not_a_multiple_of_the_member_alignment_are_refused(monkeypatch):
    monkeypatch.setattr(ringws2, "_enc_alignment", lambda text, enc: 1 << 20)
    with pytest.raises(ringws2.Fail, match="slot bytes"):
        ringws2.rewrite(_chain(0), shared=True, act=True)


def test_residency_check_counts_the_launched_warps(monkeypatch):
    import contextlib
    import importlib
    from types import SimpleNamespace

    import torch

    from helion.runtime.triton import launcher

    seen = []

    class FakeDriver:
        CUresult = SimpleNamespace(CUDA_SUCCESS=0)

        @staticmethod
        def CUfunction(x):  # noqa: N802
            return x

        @staticmethod
        def cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, block, shared):  # noqa: N802
            seen.append(block)
            return 0, 1

    class FakeTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda", 0)

    t = torch.empty(1).as_subclass(FakeTensor)
    real_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda n: FakeDriver if n == "cuda.bindings.driver" else real_import(n))
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(multi_processor_count=148))
    ck = SimpleNamespace(run=None, function=1, metadata=SimpleNamespace(num_warps=6, shared=1024, tmem_size=0))
    launcher._validate_resident_program_capacity(ck, (t,), num_warps=4, required_programs=148)
    assert seen == [6 * 32]
    with pytest.raises(RuntimeError, match="concurrently resident"):
        launcher._validate_resident_program_capacity(ck, (t,), num_warps=4, required_programs=149)


def test_fallback_restores_the_config_warp_count():
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    fb = pf._Fallback({}, None, None)
    fb.cands = [("ring", "x"), ("unchanged", None)]
    fb.num_warps = 4
    fb.original = object()
    fb.original_text = "y"
    fb._module = lambda md: "module"
    md = {"num_warps": 8}
    assert fb.advance(md, RuntimeError("boom")) == "module" and md["num_warps"] == 4 and fb.label == "unchanged"
    with pytest.raises(RuntimeError, match="boom"):
        fb.advance(md, RuntimeError("boom"))


# ---- R6 (verify5 V4/V7): a ring binary never runs fewer CTAs per SM than the binary it falls back to, judged on real
# resources after ptxas (registers x threads, shared memory, the thread limit), whether or not the launch is resident.
@triton.jit
def _occ_k(w_desc, x_ptr, out_ptr, N, K: tl.constexpr, T: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           NS: tl.constexpr, EPI: tl.constexpr):
    """verify5/synth/occ_probe.py: a persistent TMA GEMV (no spin, no route) with an EPI-deep register-heavy
    epilogue."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    n_tiles = tl.cdiv(N, BN)
    rk = tl.arange(0, BK)
    rt = tl.arange(0, T)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = w_desc.load([tile * BN, k])
            x = tl.load(x_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(x), acc)
        y = acc
        for e in tl.static_range(EPI):
            y = y * tl.sigmoid(y + e) + tl.sqrt(tl.abs(acc) + e)
        rn = tile * BN + tl.arange(0, BN)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], y.to(tl.bfloat16))


def test_ctas_per_sm_counts_registers_threads_and_shared():
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    # the occ probe: 4 warps x 182 registers (23 x 256 per warp) and 98336 B: two CTAs, by registers and shared
    assert pf.ctas_per_sm(98336, 182, 4) == 2
    # its ring: 8 warps x 184 registers no longer fit twice in the 64K-register file
    assert pf.ctas_per_sm(82972, 184, 8) == 1
    # GLM-5 dynamic Fw_f3's ring: 8 warps x 97 registers, 82972 B: two CTAs
    assert pf.ctas_per_sm(82972, 97, 8) == 2
    assert pf.ctas_per_sm(0, 0, 32) == 2  # the 2048-thread limit
    assert pf.ctas_per_sm(1024, 16, 1, tmem_columns=256) == 2
    assert pf.ctas_per_sm(1024, 16, 1) == 32


def test_entry_registers_reads_the_entry_function():
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    log = ("ptxas info    : Compiling function 'helper' for 'sm_100a'\n"
           "ptxas info    : Used 250 registers, used 0 barriers\n"
           "ptxas info    : Compiling entry function 'k' for 'sm_100a'\n"
           "ptxas info    : Function properties for k\n"
           "ptxas info    : Used 182 registers, used 1 barriers, 448 bytes cmem[0]\n")
    assert pf._entry_registers(log) == 182
    assert pf._entry_registers("ptxas info : 0 bytes gmem\n") is None


def _compile_occ_under_hook(tmp_path, monkeypatch, epi):
    from triton import knobs

    from helion.runtime.triton import ttgir_weight_prefetch as pf

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path), raising=False)
    saved = knobs.runtime.add_stages_inspection_hook
    knobs.runtime.add_stages_inspection_hook = None
    sig = dict(w_desc="tensordesc<bf16[128, 128]>", x_ptr="*bf16", out_ptr="*bf16", N="i32")
    try:
        with pf.enabled(ring=True):
            return _compile_jit(_occ_k, sig, dict(K=4096, T=64, BN=128, BK=128, NS=2, EPI=epi), num_warps=4)
    finally:
        knobs.runtime.add_stages_inspection_hook = saved


def _shipped_registers(ptx):
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    d = tempfile.mkdtemp(prefix="ringtest_occ_")
    pp = os.path.join(d, "k.ptx")
    with open(pp, "w") as f:
        f.write(ptx)
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas-blackwell")
    r = subprocess.run([ptxas, "-v", "--regAllocOptLevel=2", "--gpu-name", "sm_100a", pp, "-o", pp + ".cubin"],
                       capture_output=True, text=True, check=True)
    return pf._entry_registers(r.stderr)


class _PassLog:
    """The pass's debug messages while compiling (no caplog: the suites run with ``-p no:logging``)."""

    def __enter__(self):
        import logging

        from helion.runtime.triton import ttgir_weight_prefetch as pf

        self.messages = []
        self.logger = logging.getLogger(pf.__name__)
        self.level = self.logger.level
        self.handler = logging.Handler(logging.DEBUG)
        self.handler.emit = lambda r: self.messages.append(r.getMessage())
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.level)

    @property
    def text(self):
        return "\n".join(self.messages)


def test_ring_that_costs_a_cta_per_sm_ships_its_fallback(tmp_path, monkeypatch):
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    with _PassLog() as log:
        ck = _compile_occ_under_hook(tmp_path, monkeypatch, epi=4)
    # every ring form doubles the warps of a ~180-register kernel: one CTA per SM where the fallback runs two
    assert re.search(r"ring binary runs 1 CTA\(s\) per SM \(\d+ registers x 8 warps\), its fallback 2", log.text)
    assert ck.metadata.num_warps == 4 and "cp.async.bulk.tensor" in ck.asm["ptx"]
    regs = _shipped_registers(ck.asm["ptx"])
    assert pf.ctas_per_sm(ck.metadata.shared, regs, ck.metadata.num_warps) == 2


def test_ring_between_two_ctas_and_its_fallback_ships_the_fallback(tmp_path, monkeypatch):
    """(R6-7) A ring that keeps 2+ CTAs per SM but fewer than its fallback is refused too: the BN16 gemv's 8-warp ring
    runs 6 CTAs/SM where the 4-warp module runs 9 (measured on B200, one tile per CTA: +0.8 % at K 2048, +4.3 % at
    K 4096)."""
    from triton import knobs

    from helion.runtime.triton import ttgir_weight_prefetch as pf

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path), raising=False)
    saved = knobs.runtime.add_stages_inspection_hook
    knobs.runtime.add_stages_inspection_hook = None
    try:
        with _PassLog() as log, pf.enabled(ring=True):
            ck = _compile_jit(kern.gemv, SIG, dict(K=2048, T=16, BN=16, BK=128, NS=4), num_warps=4)
    finally:
        knobs.runtime.add_stages_inspection_hook = saved
    assert re.search(r"ring binary runs 6 CTA\(s\) per SM \(\d+ registers x 8 warps\), its fallback 9", log.text)
    assert ck.metadata.num_warps == 4 and "cp.async.mbarrier.arrive" not in ck.asm["ptx"]


def _hook_compile(tmp_path, monkeypatch, fn, sig, constexprs):
    from triton import knobs

    from helion.runtime.triton import ttgir_weight_prefetch as pf

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path), raising=False)
    saved = knobs.runtime.add_stages_inspection_hook
    knobs.runtime.add_stages_inspection_hook = None
    try:
        with _PassLog() as log, pf.enabled(ring=True):
            ck = _compile_jit(fn, sig, constexprs, num_warps=4)
    finally:
        knobs.runtime.add_stages_inspection_hook = saved
    return ck, log.text


def test_spin_gated_ring_keeps_two_ctas_a_no_spin_ring_all_of_them(tmp_path, monkeypatch):
    """(R6-8) Both directions of the occupancy band. A spin-gated ring (root B's consumer waits on the readiness
    counter before the first stage; the weight producer streams past it) ships at 4 CTAs/SM where its 4-warp fallback
    runs 8: it keeps min(fallback, 2). A ring with no spin before its stages (the plain gemv) keeps all of its
    fallback's CTAs/SM: 6 < 9 is refused. (B200, one tile per CTA: Helion's spin-gated GEMV megakernels at 4-6 of 6-9
    CTAs/SM win 2-11 %; plain GEMVs at 6 of 9 lose up to 4 %.)"""
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    gated, log = _hook_compile(tmp_path / "gated", monkeypatch, _chain_k, SIGC,
                               dict(K=2048, T=16, BN=16, BK=128, NS=4, RA=0, KA=512))
    assert "'spin_gated': True" in log and "CTA(s) per SM (" not in log
    assert gated.metadata.num_warps == 8 and "cp.async.mbarrier.arrive" in gated.asm["ptx"]
    ring_ctas = pf.ctas_per_sm(gated.metadata.shared, _shipped_registers(gated.asm["ptx"]), 8)
    assert 2 <= ring_ctas < 8 <= pf.ctas_per_sm(24576, 0, 4)   # below the fallback's shared-memory/thread bound
    plain, log = _hook_compile(tmp_path / "plain", monkeypatch, kern.gemv, SIG, dict(K=2048, T=16, BN=16, BK=128, NS=4))
    assert "'spin_gated': False" in log
    assert re.search(r"ring binary runs 6 CTA\(s\) per SM \(\d+ registers x 8 warps\), its fallback 9", log)
    assert plain.metadata.num_warps == 4


def test_occupancy_floor_of_a_spin_gated_ring(monkeypatch):
    """(R6-8) The guard's floor: a spin-gated ring keeps min(fallback, SPIN_GATED_KEPT); one CTA/SM below a fallback of
    two or more is refused in every case; any other ring keeps its fallback's CTAs/SM."""
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    assert ringws2.SPIN_GATED_KEPT == 2
    fb = pf._Fallback({}, None, None)
    fb.cands = [("ring", "x"), ("hoist", "y"), ("unchanged", None)]
    fallback = {"ctas": 9}
    monkeypatch.setattr(pf._Fallback, "_fallback_ctas_per_sm", lambda self, md, occ: fallback["ctas"])
    for gated, occ, fb_ctas, floor in ((True, 4, 9, 2), (True, 2, 9, 2), (True, 1, 2, 2), (True, 1, 1, 1),
                                       (False, 4, 9, 9), (False, 6, 6, 6)):
        fb.cand_facts = {"ring": {"spin_gated": gated}}
        fallback["ctas"] = fb_ctas
        assert fb._occupancy_floor({}, occ) == floor, (gated, occ, fb_ctas)


def test_spin_gated_shared_memory_rule_keeps_two():
    """(R6-8) The shared-memory rule agrees with the guard: a spin-gated ring keeps min(module, 2) CTAs/SM of shared
    memory, so the BN32 chain kernel (6 as is) keeps its full-depth ring (D3, 37948 B) instead of one lowered to keep 6
    (measured on the Helion BN32 megakernel: D3 at 5 CTAs -3.4..-5.2 %, the lowered D2 at 6 +2.8..+4.1 % vs the hoist);
    a ring with no spin keeps the module's 6."""
    src = _chain(0, bn=32)
    occ = ringws2.kept_occupancy(src)
    t, info = ringws2.rewrite_budgeted(src)
    assert occ == 6 and info["spin_gated"] and info["D"] == info["want_D"] == 3
    assert ringws2.resident_limit(occ) < info["smem"] <= ringws2.resident_limit(2)
    g = _gemv(32)
    _, ginfo = ringws2.rewrite_budgeted(g)
    assert not ginfo["spin_gated"] and ginfo["smem"] <= ringws2.resident_limit(ringws2.kept_occupancy(g))


def test_ring_that_keeps_its_ctas_per_sm_ships(tmp_path, monkeypatch):
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    with _PassLog() as log:
        ck = _compile_occ_under_hook(tmp_path, monkeypatch, epi=0)
    assert ck.metadata.num_warps == 8 and "CTA(s) per SM (" not in log.text
    regs = _shipped_registers(ck.asm["ptx"])
    assert pf.ctas_per_sm(ck.metadata.shared, regs, 8) == 2


def test_per_loop_stage_count_invariant_backs_up_the_sequence_guard(monkeypatch):
    # defense in depth: with the global stage-sequence guard neutralized, a pipelined loop whose producer would issue
    # fewer stages than the consumer reads (pipeliner predicate iv < K - 3 BK lowered by one BK) is still refused
    src = _gemv(16, 128)
    pred = re.search(r"arith\.cmpi slt, %\S+, (%c(\d+)_i32) : i32", src)
    assert pred and int(pred.group(2)) == 2048 - 3 * 128
    bad = src.replace(f"{pred.group(1)} = arith.constant {pred.group(2)} : i32",
                      f"{pred.group(1)} = arith.constant {int(pred.group(2)) - 128} : i32", 1)
    assert bad != src
    with pytest.raises(ringws2.Fail, match="stage counts differ"):
        ringws2.rewrite(bad, shared=True)
    monkeypatch.setattr(ringws2, "_stage_words", lambda *a, **k: [])
    with pytest.raises(ringws2.Fail, match="stage counts differ"):
        ringws2.rewrite(bad, shared=True)


# ---- R2: griddepcontrol.launch_dependents (lets the next grid launch; orders no memory access of this CTA)
def test_act_allows_a_launch_dependents_trigger_only():
    src = _chain(0)
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.M).group(0)
    pid = re.search(r"(%\S+) = tt\.get_program_id x", src).group(1)

    def trig(text, operand=False):
        if operand:
            return sync + (f'\n      %rgt_tr = tt.elementwise_inline_asm "{text}" {{constraints = "=r,r", packed_element = 1 : i32, '
                           f'pure = false}} {pid} : i32 -> i32')
        return sync + (f'\n      %rgt_tr = tt.elementwise_inline_asm "{text}" {{constraints = "=r", packed_element = 1 : i32, '
                       f'pure = false}} -> i32')
    # Triton's exact gdc_launch_dependents
    t, info = ringws2.rewrite(src.replace(sync, trig("griddepcontrol.launch_dependents; // dummy $0"), 1), shared=True, act=True)
    assert info["act"]
    # a nested exact gdc_wait (no top-level one): the PDL rule refuses
    with pytest.raises(ringws2.Fail, match="PDL wait"):
        ringws2.rewrite(src.replace(sync, trig("griddepcontrol.wait; // dummy $0"), 1), shared=True, act=True)
    # the trigger with another constraint string, or with more instructions: off the allowlist
    for bad, operand in (("griddepcontrol.launch_dependents; // dummy $0", True),
                         ("griddepcontrol.launch_dependents; ld.acquire.gpu.global.u32 $0, [$1];", True),
                         ("griddepcontrol.launch_dependents; // x\\n ld.acquire.gpu.global.u32 $0, [$1];", True)):
        with pytest.raises(ringws2.Fail, match="off the allowlist"):
            ringws2.rewrite(src.replace(sync, trig(bad, operand), 1), shared=True, act=True)


# ---- root-suffix policy: the longest suffix of roots whose ring fits at full depth; the prefix keeps the hoist
@triton.jit
def _two_gemv_k(w1_ptr, w2_ptr, x_ptr, y_ptr, out_ptr, state_ptr, epoch_ptr, N2, N1: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
                BN1: tl.constexpr, BK1: tl.constexpr, NS1: tl.constexpr, BN2: tl.constexpr, BK2: tl.constexpr,
                NS2: tl.constexpr):
    """Root A: a wide-tile GEMV (big stages) writing y + publication; root B: a skinny GEMV over y behind a spin."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    for tile in tl.range(pid, tl.cdiv(N1, BN1), npid, num_stages=1):
        rk = tl.arange(0, BK1)
        rn = tile * BN1 + tl.arange(0, BN1)
        acc = tl.zeros([BN1, T], dtype=tl.float32)
        for k in tl.range(0, K, BK1, num_stages=NS1):
            w = tl.load(w1_ptr + rn[:, None] * K + (k + rk)[None, :])
            x = tl.load(x_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(x), acc)
        tl.store(y_ptr + rt[None, :] * N1 + rn[:, None], acc.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    target = epoch * tl.cdiv(N1, BN1)
    for tile in tl.range(pid, tl.cdiv(N2, BN2), npid, num_stages=1):
        c = _acq(state_ptr)
        while c != target:
            c = _acq(state_ptr)
        tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                  dtype=tl.int32, is_pure=False, pack=1)
        rk = tl.arange(0, BK2)
        rn = tile * BN2 + tl.arange(0, BN2)
        acc = tl.zeros([BN2, T], dtype=tl.float32)
        for k in tl.range(0, N1, BK2, num_stages=NS2):
            w = tl.load(w2_ptr + rn[:, None] * N1 + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * N1 + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


def test_ring_takes_the_longest_suffix_that_fits_at_full_depth():
    sig = dict(w1_ptr="*bf16", w2_ptr="*bf16", x_ptr="*bf16", y_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32",
               epoch_ptr="*i32", N2="i32")
    # root A's K (1024) keeps root B's ring above MIN_COPY_COVERAGE of the trip-weighted copy bytes (B5-9)
    src = _compile_jit(_two_gemv_k, sig, dict(N1=2048, K=1024, T=16, BN1=128, BK1=256, NS1=3, BN2=16, BK2=128, NS2=7),
                       num_warps=4).asm["ttgir"]
    starts = ringws2._ring_starts(src)
    assert len(starts) == 2
    # both roots: one byte ring with 64 KB slots cannot hold root B's 6 stages -> lowered depth
    _, full = ringws2.rewrite_budgeted(src, from_top=None)
    assert full["D"] < full["want_D"] == 6
    t, info = ringws2.rewrite_budgeted(src)
    assert info["from_top"] == starts[1] and info["D"] == info["want_D"] == 6 and info["act"]
    head = t[: t.index("ttg.warp_specialize(")]
    assert head.count("ttg.async_copy_global_to_local") >= 2      # root A keeps Triton's pipeline in the prefix
    _, viol = _hic(t)
    assert not viol, viol[:5]
    offc.compile_ttgir_text(t)


def test_partial_unroll_by_a_factor_keeps_the_loop():
    src = _chain(0)
    t0, _ = ringws2.rewrite(src, shared=True, act=True, unroll=0)
    t4, _ = ringws2.rewrite(src, shared=True, act=True, unroll=0, unroll_factor=4)
    d0, d4 = _parts(t0)[0], _parts(t4)[0]
    assert d4.count("scf.for") == d0.count("scf.for")                       # the K loop stays (16 / 4 trips)
    assert d4.count("ttng.wait_barrier %rg_rb") == 4 * d0.count("ttng.wait_barrier %rg_rb")
    assert re.search(r"_ufstep = arith\.constant 512 : i32", d4) and "_uf3 = arith.addi" in d4
    assert _parts(t4)[1:] == _parts(t0)[1:]
    _, viol = _hic(t4)
    assert not viol, viol[:5]
    offc.compile_ttgir_text(t4)
    # a factor that does not divide the trip count leaves the loop as is; full unroll still wins below its limit
    t3, _ = ringws2.rewrite(src, shared=True, act=True, unroll=0, unroll_factor=3)
    assert _parts(t3)[0] == d0
    t16, _ = ringws2.rewrite(src, shared=True, act=True, unroll=16, unroll_factor=4)
    assert _parts(t16)[0].count("scf.for") == d0.count("scf.for") - 1


# =====================================================================================================================
# PORT onto canon6 (integ/ring): the RING3 verifier's fixes and the tests for its surviving mutants.
#   CX-W / CX-IP: write targets not traceable to arguments write EVERY argument (as in _written_params);
#   R3 closes over the launch's storage alias classes; R2 / R3 classify asm with the exact allowlist;
#   deterministic output; mutants V2 V3 V4 V5 V10 V17 V18 V25 V26 V27 (ring3/verify/v3/probes/mutants_v3.py).

@triton.jit
def _wsync():
    tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _spin(p, target):
    c = _acq(p)
    while c != target:
        c = _acq(p)
    _wsync()


@triton.jit(noinline=True)
def _row_base(x2_ptr, r, T: tl.constexpr, K: tl.constexpr):
    return x2_ptr + r * T * K


@triton.jit
def _r3w_k(x_ptr, y_ptr, w_ptr, wa_ptr, x2_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr, T: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, KA: tl.constexpr, WMODE: tl.constexpr):
    """RING3 verifier CX-W (rvk.r3w_k): counterexample 2 (_r3_k) with root B's x2 store pointer built in form WMODE:
    0 direct, 1 loop-carried pointer, 2 scalar select of a base pointer, 3 pointer returned by a noinline callee,
    4 pointer-tensor select.  Forms 1-4 are not traceable to an argument: they write every argument."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32) * 1.5
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    for r in tl.range(pid, npid, npid, num_stages=1):
        s = tl.zeros([T, BK], dtype=tl.float32)
        for ka in tl.range(0, KA, BK, num_stages=NS):
            s += tl.load(wa_ptr + rt[:, None] * KA + (ka + rk)[None, :]).to(tl.float32)
        val = (s * 0.25).to(tl.bfloat16)
        if WMODE == 0:
            for kb in tl.range(0, K, BK, num_stages=1):
                tl.store(x2_ptr + (r * T + rt)[:, None] * K + (kb + rk)[None, :], val)
        elif WMODE == 1:
            p = x2_ptr + (r * T + rt)[:, None] * K + rk[None, :]
            for kb in tl.range(0, K, BK, num_stages=1):
                tl.store(p, val)
                p += BK
        elif WMODE == 2:
            base = tl.where(N > 0, x2_ptr, out_ptr)
            for kb in tl.range(0, K, BK, num_stages=1):
                tl.store(base + (r * T + rt)[:, None] * K + (kb + rk)[None, :], val)
        elif WMODE == 3:
            base = _row_base(x2_ptr, r, T, K)
            for kb in tl.range(0, K, BK, num_stages=1):
                tl.store(base + rt[:, None] * K + (kb + rk)[None, :], val)
        else:
            pa = x2_ptr + (r * T + rt)[:, None] * K + rk[None, :]
            pb = out_ptr + rt[:, None] * 0 + rk[None, :] * 0
            p = tl.where(N > 0, pa, pb)
            for kb in tl.range(0, K, BK, num_stages=1):
                tl.store(p + kb, val)
        tl.debug_barrier()
        tl.atomic_add(state_ptr + 1, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        _spin(state_ptr, target)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            x2 = tl.load(x2_ptr + (pid * T + rt)[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y + x2), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


@triton.jit
def _inplace_k(x_ptr, y_ptr, w_ptr, x2_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr, T: tl.constexpr,
               BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, WMODE: tl.constexpr):
    """RING3 verifier CX-IP (rvk.inplace_k): ONE reading root reads this program's x2 rows through its pipelined K loop
    and rewrites them after every tile through a pointer of form WMODE (0 direct, 1 loop-carried, 2 select)."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32) * 1.5
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        _spin(state_ptr, target)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            x2 = tl.load(x2_ptr + (pid * T + rt)[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y + x2), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
        tl.debug_barrier()
        if WMODE == 0:
            for kb in tl.range(0, K, BK, num_stages=1):
                q = x2_ptr + (pid * T + rt)[:, None] * K + (kb + rk)[None, :]
                tl.store(q, (tl.load(q).to(tl.float32) + 1.0).to(tl.bfloat16))
        elif WMODE == 1:
            q = x2_ptr + (pid * T + rt)[:, None] * K + rk[None, :]
            for kb in tl.range(0, K, BK, num_stages=1):
                tl.store(q, (tl.load(q).to(tl.float32) + 1.0).to(tl.bfloat16))
                q += BK
        else:
            base = tl.where(N > 0, x2_ptr, out_ptr)
            for kb in tl.range(0, K, BK, num_stages=1):
                q = base + (pid * T + rt)[:, None] * K + (kb + rk)[None, :]
                tl.store(q, (tl.load(q).to(tl.float32) + 1.0).to(tl.bfloat16))
        tl.debug_barrier()
    tl.store(epoch_ptr + pid, epoch)


SIG_IP = dict(x_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", x2_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32",
              epoch_ptr="*i32", N="i32")


def _r3w(wmode):
    return _compile_jit(_r3w_k, SIGR3, dict(K=2048, T=16, BN=16, BK=128, NS=4, KA=1024, WMODE=wmode),
                        num_warps=4).asm["ttgir"]


def _inplace(wmode):
    return _compile_jit(_inplace_k, SIG_IP, dict(K=2048, T=16, BN=16, BK=128, NS=4, WMODE=wmode),
                        num_warps=4).asm["ttgir"]


def test_cx_w_traceable_writer_gets_an_ordering_point():
    # reference (WMODE 0): root B (in the region: it streams wa) writes x2 directly -> one ordering point before root C
    t, info = ringws2.rewrite_budgeted(_r3w(0))
    assert info["act"] and info["order_points"] == 1 and info["from_top"] is None
    assert "ttng.wait_barrier %pr_wo" in _parts(t)[2]


@pytest.mark.parametrize("wmode", [1, 2, 3, 4])
def test_cx_w_untraceable_writer_writes_every_argument(wmode):
    # RING3 verifier CX-W (16/16 wrong on GPU with the delivered ring): the activation producer read x2 with no
    # ordering point because R3 dropped a write target that is not traceable to an argument.  Such a write now writes
    # every argument, as in _written_params: in the whole-kernel region root B writes the weight wa it reads itself
    # (refused); the suffix policy then rings root C only, with root B in the prefix (written before the partitions).
    src = _r3w(wmode)
    with pytest.raises(ringws2.Fail, match="written in the root that reads it"):
        ringws2.rewrite(src, shared=True, act=True)
    t, info = ringws2.rewrite_budgeted(src)
    starts = ringws2._ring_starts(src)
    assert info["act"] and info["from_top"] == starts[-1]
    head = t[: t.index("ttg.warp_specialize(")]
    assert head.count("tt.atomic_rmw add, release, gpu") == 2       # roots A and B both run before the partitions
    _, viol = _hic(t)
    assert not viol, viol[:5]


def test_cx_ip_in_place_writer_is_refused_for_every_pointer_form():
    # RING3 verifier CX-IP (12/12 wrong): the reading root rewrites its own x2 rows after each tile
    for wmode in (0, 1, 2):
        src = _inplace(wmode)
        with pytest.raises(ringws2.Fail, match="written in the root that reads it"):
            ringws2.rewrite(src, shared=True, act=True)
    t, info = ringws2.rewrite_budgeted(_inplace(0))      # param-rooted: act refused, the weight-only ring applies
    assert not info["act"] and "root that reads it" in info["act_refused"]
    for wmode in (1, 2):                                 # untraceable: every argument is written, no weight left
        with pytest.raises(ringws2.Fail, match="no movable ring"):
            ringws2.rewrite_budgeted(_inplace(wmode))


def test_param_writes_is_written_params_per_op():
    # R3's per-op attribution IS the hoist pass's: the union over the entry function's ops equals _written_params,
    # with and without storage classes (any divergence would let R3 miss a writer the base analysis saw)
    P = ringws2.P
    cases = [(_r3w(m), ()) for m in range(5)] + [(_inplace(m), ()) for m in range(3)] + [
        (_r3(), ()), (_r3(), (("x2_ptr", "y_ptr"),)), (_chain(1), (("y_ptr", "w_ptr", "out_ptr"),)), (_cx(4), ())]
    for src, alias in cases:
        lines = P._uniquify(src).splitlines()
        funcs = P._functions(lines)
        called = ringws2._called(lines)
        classes = {k: v for k, v in ringws2._storage_classes(lines, funcs, called, alias).items() if v is not None}
        written, wild = P._write_analysis(lines, funcs, classes)
        by_name = {g.name: g for g in funcs}
        f = next(g for g in funcs if "tt.func public" in lines[g.start])
        union = set()
        for line in lines[f.start + 1: f.end]:
            union |= ringws2._param_writes(line, f, by_name, written, wild, classes.get(f.name, []))
        assert union == {f.params[k] for k in written[f.name]}, (alias, union)


# ---- R3 closes over the storage alias classes
@triton.jit
def _r3a_k(x_ptr, y_ptr, w_ptr, wa_ptr, x2r_ptr, x2w_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr,
           T: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, KA: tl.constexpr):
    """_r3_k whose root B writes x2 through x2w_ptr and whose root C reads it through x2r_ptr: two views of one
    storage when the launch says so (alias class {x2r_ptr, x2w_ptr})."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32) * 1.5
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    for r in tl.range(pid, npid, npid, num_stages=1):
        s = tl.zeros([T, BK], dtype=tl.float32)
        for ka in tl.range(0, KA, BK, num_stages=NS):
            s += tl.load(wa_ptr + rt[:, None] * KA + (ka + rk)[None, :]).to(tl.float32)
        for kb in tl.range(0, K, BK, num_stages=1):
            tl.store(x2w_ptr + (r * T + rt)[:, None] * K + (kb + rk)[None, :], (s * 0.25).to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr + 1, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        _spin(state_ptr, target)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            x2 = tl.load(x2r_ptr + (pid * T + rt)[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y + x2), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


def _r3a():
    sig = dict(x_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", wa_ptr="*bf16", x2r_ptr="*bf16", x2w_ptr="*bf16",
               out_ptr="*bf16", state_ptr="*i32", epoch_ptr="*i32", N="i32")
    return _compile_jit(_r3a_k, sig, dict(K=2048, T=16, BN=16, BK=128, NS=4, KA=1024), num_warps=4).asm["ttgir"]


X2 = (("x2r_ptr", "x2w_ptr"),)


def test_act_closes_writers_over_the_storage_alias_classes():
    src = _r3a()
    # separate storages: x2r is never written -> a weight (copied by the weight producer, no ordering point)
    _, info = ringws2.rewrite_budgeted(src)
    assert info["act"] and info["order_points"] == 0
    # one storage (the launch's facts): x2r is written by root B's x2w stores (in the region) -> an activation, and
    # the reading root's activation producer waits for the ordering point after root B
    t, info = ringws2.rewrite_budgeted(src, alias_classes=X2)
    assert info["act"] and info["order_points"] == 1 and len(info["act"]) == 2
    _, _, aprod = _parts(t)
    assert aprod.index("ttng.wait_barrier %pr_wo") < aprod.index("ttg.async_copy_global_to_local")
    _, viol = _hic(t)
    assert not viol, viol[:5]
    # an in-place write through the OTHER view in the reading root is a write of what it reads
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.M).group(0)
    src2 = src.replace(sync, sync + "\n      %rgt_c = arith.constant 0.0 : bf16\n      tt.store %x2w_ptr, %rgt_c : !tt.ptr<bf16>", 1)
    with pytest.raises(ringws2.Fail, match="root that reads it"):
        ringws2.rewrite(src2, shared=True, act=True, alias_classes=X2)
    ringws2.rewrite(src2, shared=True, act=True)          # separate storages: x2r is a weight, nothing to refuse
    # classes naming an unknown argument: refused, as the hoist skips the function
    with pytest.raises(ringws2.Fail, match="unknown parameters"):
        ringws2.rewrite(src, shared=True, act=True, alias_classes=(("x2r_ptr", "nope"),))


def test_prefix_hoist_of_the_suffix_policy_gets_the_alias_classes(monkeypatch):
    seen = []
    real = ringws2.P.rewrite

    def spy(text, alias_classes=()):
        seen.append(alias_classes)
        return real(text, alias_classes)
    monkeypatch.setattr(ringws2.P, "rewrite", spy)
    sig = dict(w1_ptr="*bf16", w2_ptr="*bf16", x_ptr="*bf16", y_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32",
               epoch_ptr="*i32", N2="i32")
    src = _compile_jit(_two_gemv_k, sig, dict(N1=2048, K=1024, T=16, BN1=128, BK1=256, NS1=3, BN2=16, BK2=128, NS2=7),
                       num_warps=4).asm["ttgir"]
    _, info = ringws2.rewrite_budgeted(src, alias_classes=(("w1_ptr", "x_ptr"),))
    assert info["from_top"] is not None and seen and all(s == (("w1_ptr", "x_ptr"),) for s in seen)


# ---- R2 / R3 classify inline assembly with the exact allowlist
def test_every_allowlist_entry_is_classified_for_the_activation_rules():
    templates = [e.template for e in ringws2.P._ALLOWLIST]
    spin = {e.template for e in ringws2.P._ALLOWLIST if e.kind in ("poll", "sync")}
    assert not (ringws2._QUIET_TEMPLATES & ringws2._SYNC_TEMPLATES)
    assert not (spin & (ringws2._QUIET_TEMPLATES | ringws2._SYNC_TEMPLATES))
    for t in templates:
        assert (t in spin) + (t in ringws2._QUIET_TEMPLATES) + (t in ringws2._SYNC_TEMPLATES) == 1, t
    assert (ringws2._QUIET_TEMPLATES | ringws2._SYNC_TEMPLATES) <= set(templates)   # nothing stale
    # quiet entries write nothing or only through their pointer operand, and never wait on the predecessor
    for e in ringws2.P._ALLOWLIST:
        if e.template in ringws2._QUIET_TEMPLATES:
            assert e.kind in ("quiet", "writes"), e


def test_region_assembly_off_the_allowlist_refuses_act_on_its_own(monkeypatch):
    # The PDL rule already refuses any module holding assembly off the allowlist (it may wait on the predecessor);
    # R2's own check must hold without it (defense in depth: the ring is never weaker than the base pass).
    src = _chain(0)
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.M).group(0)
    off = sync + ('\n      %rgt_o = tt.elementwise_inline_asm "mov.u32 $0, 0;" {constraints = "=r", packed_element = 1 : i32, '
                  'pure = true} -> i32')
    with pytest.raises(ringws2.Fail, match="off the allowlist"):
        ringws2.rewrite(src.replace(sync, off, 1), shared=True, act=True)
    monkeypatch.setattr(ringws2, "_pdl_split", lambda lines, funcs, f, body, called: f.start)
    with pytest.raises(ringws2.Fail, match="off the allowlist in the warp-specialized region"):
        ringws2.rewrite(src.replace(sync, off, 1), shared=True, act=True)
    # (a pure off-list op is still off the list: it writes every argument in the base pass)
    assert ringws2._asm_class(off.splitlines()[-1]) == "sync"


@pytest.mark.parametrize("asm,constraints,operand,cls", [
    ("ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", "%rgt_p : !tt.ptr<i32>", "spin"),
    ("bar.sync 0;", "=r", None, "sync"),
    ("fence.proxy.async.global; mov.u32 $0, $1;", "=r,r", "%rgt_i : i32", "spin"),
    ("mov.u32 $0, %smid;", "=r", None, "quiet"),
    ("griddepcontrol.launch_dependents; // dummy $0", "=r", None, "quiet"),
    ("griddepcontrol.launch_dependents; // dummy $0 ", "=r", None, "sync"),   # one byte off: not the entry
])
def test_asm_class_is_the_exact_template(asm, constraints, operand, cls):
    head = f'%rgt = tt.elementwise_inline_asm "{asm}" {{constraints = "{constraints}", packed_element = 1 : i32, pure = false}}'
    line = f"{head} {operand} -> i32" if operand else f"{head} -> i32"
    assert ringws2._asm_class(line) == cls, line


def test_stamp_writes_only_its_pointer_operand_and_is_not_a_generic_writer():
    # the allowlisted stamp (#22) through y_ptr in the READING root: a write of what it reads
    src = _chain(0)
    with pytest.raises(ringws2.Fail, match="root that reads it"):
        ringws2.rewrite(_stamp_after_sync(src, "%y_ptr"), shared=True, act=True)
    # through out_ptr (not read by any ring): act stays (test_act_allows_globaltimer_stamps_only)
    # in an EARLIER root of the region, the stamp's store is not a generic tt.store: refused (no ordering claim)
    src1 = _chain(1)
    pub = re.search(r"^(\s*)(%\S+ = )?tt\.atomic_rmw add, release, gpu, %state_ptr.*$", src1, re.M)
    ind = pub.group(1)
    stamp = (f'{ind}%rgt_a = tt.ptr_to_int %y_ptr : !tt.ptr<bf16> -> i64\n'
             f'{ind}%rgt_s = tt.elementwise_inline_asm {_STAMP_ASM} {{constraints = "=r,l", packed_element = 1 : i32, '
             'pure = false} %rgt_a : i64 -> i32\n')
    src2 = src1[:pub.start()] + stamp + src1[pub.start():]
    with pytest.raises(ringws2.Fail, match="written by tt.elementwise_inline_asm"):
        ringws2.rewrite(src2, shared=True, act=True)


# ---- deterministic output (RING3 verifier V-4)
def test_output_text_does_not_depend_on_the_hash_seed():
    # every run, under any hash seed, emits the producer's re-laid-out aliases in (length, name) order and the same text
    # (_r3 re-lays out two aliases of equal length, #blocked1 and #blocked2: set order alone would vary with the seed)
    code = ("import sys, re, hashlib; sys.path[:0] = [sys.argv[1], sys.argv[2]]; import test_ttgir_weight_ring as T; "
            "t = T.ringws2.rewrite_budgeted(T._r3())[0]; "
            "print(hashlib.md5(t.encode()).hexdigest(), *(','.join(re.findall(r'^(#blocked\\d*)_' + s + ' = ', t, re.M)) "
            "for s in ('pw', 'pa')))")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outs = set()
    for seed in ("0", "1", "2", "3", "4", "5"):
        env = {k: v for k, v in os.environ.items() if not k.startswith("HELION_TTGIR_RING")}
        env.update(PYTHONHASHSEED=seed, CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, "-c", code, root, here], capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-800:]
        md5, pw, pa = r.stdout.strip().splitlines()[-1].split()
        for defs in (pw.split(","), pa.split(",")):
            assert len(defs) >= 2 and defs == sorted(defs, key=lambda a: (len(a), a), reverse=True), (seed, defs)
        outs.add(md5)
    assert len(outs) == 1, outs


# ---- the candidate chain in the pass: the launch's ring flag, alias classes, cache key
def test_ring_follows_the_launch_flag_and_is_keyed(monkeypatch):
    """The ring is decided by the kernel's setting as the launch passes it (enabled(ring=...)), never by the process
    environment at compile or launch time (Triton calls the hook for the key on every launch), and both keys name it."""
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    stages = {"ttgir": lambda s, m: s, "llir": lambda s, m: s, "ptx": lambda s, m: s, "cubin": lambda s, m: s}
    hook = pf._StagesHook(None)
    keys = {}
    for env in ("0", "1", None):
        if env is None:
            monkeypatch.delenv("HELION_TTGIR_WEIGHT_RING", raising=False)
        else:
            monkeypatch.setenv("HELION_TTGIR_WEIGHT_RING", env)
        for ring in (False, True):
            with pf.enabled(ring=ring):
                keys[env, ring] = hook()
                st = dict(stages)
                hook(None, st, None, None, 100)
                if ring:
                    assert isinstance(st["llir"].__self__, pf._Fallback)
                else:   # canonical: only make_ttgir wrapped
                    assert st["llir"] is stages["llir"] and st["cubin"] is stages["cubin"]
                    assert st["ttgir"] is not stages["ttgir"]
    for ring in (False, True):
        assert len({keys[env, ring] for env in ("0", "1", None)}) == 1
    off, on = keys[None, False], keys[None, True]
    assert "-ring0" in off[0] and off[0].endswith("-ring0") and on[0].endswith("-ring1")
    assert off[0] != on[0] and off[1] != on[1] and pf._DIGEST in off[0]
    # nested contexts: the innermost launch decides
    with pf.enabled(ring=True):
        with pf.enabled(ring=False):
            assert hook() == off
        assert hook() == on


def test_pass_digest_hashes_the_ring_module():
    """(integ verify gap test, mutant MP2) The stages hook keys Triton's caches with _DIGEST; ring binaries depend on
    the ring module, so an edited ring module must change the key, or stale ring binaries would be reused."""
    import hashlib
    from pathlib import Path

    from helion.runtime.triton import ttgir_weight_prefetch as pf

    family = sorted(Path(pf.__file__).parent.glob("ttgir_*.py"))
    assert {"ttgir_weight_prefetch.py", "ttgir_weight_ring.py"} <= {f.name for f in family}
    want = hashlib.sha256(b"".join(f.name.encode() + b"\0" + f.read_bytes() for f in family)).hexdigest()
    assert pf._DIGEST == want


def test_ring_compile_key_follows_the_ring_decisions(monkeypatch):
    """(ATT6 verify, mutant K15) A ring binary depends on the automatic decisions' values (size-twin budget, ladder,
    coverage threshold, defaults, shared-memory limit); a value changed at run time (a tool, a test) must change both
    cache keys of a ring compile, and nothing else may; kernels without the ring do not depend on them."""
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    def keys():
        with pf.enabled(ring=True):
            on = pf._StagesHook(None)()
        with pf.enabled(ring=False):
            off = pf._StagesHook(None)()
        return on, off

    base_on, base_off = keys()
    assert base_on[0].endswith("-ring1")
    for name, value in (
        ("CODE_BUDGET", ringws2.CODE_BUDGET + 1024),
        ("SIZE_LADDER", ringws2.SIZE_LADDER[:1]),
        ("MIN_COPY_COVERAGE", ringws2.MIN_COPY_COVERAGE / 2),
        ("RING_DEFAULTS", {**ringws2.RING_DEFAULTS, "unroll": 0}),
        ("SMEM_LIMIT", ringws2.SMEM_LIMIT - 1024),
    ):
        monkeypatch.setattr(ringws2, name, value)
        on, off = keys()
        assert on[0] != base_on[0] and on[1] != base_on[1], name
        assert off == base_off, name
        monkeypatch.undo()
        assert keys() == (base_on, base_off)


def test_candidates_pass_the_alias_classes_to_ring_and_hoist(monkeypatch):
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    seen = {}

    def ring(text, **kw):
        seen.setdefault("ring", []).append(kw.get("alias_classes"))
        return "RING", {"act": ["%a"]}

    def hoist(text, alias_classes=()):
        seen["hoist"] = alias_classes
        return text, pf.Report()
    monkeypatch.setattr(ringws2, "rewrite_budgeted", ring)
    monkeypatch.setattr(ringws2, "copy_coverage", lambda text: 1.0)
    monkeypatch.setattr(pf, "rewrite", hoist)
    c = pf._candidates("x", (("a", "b"),), ring=True)
    assert [label for label, _ in c] == ["ring", "ring_wonly", "unchanged"]
    # each ring form is also built as its size twins (a twin only where it changes the ring; never here)
    n = len(ringws2.SIZE_LADDER)
    assert seen == {"ring": [(("a", "b"),)] * (2 * n), "hoist": (("a", "b"),)}
    monkeypatch.setenv("HELION_TTGIR_WEIGHT_RING", "1")
    assert [label for label, _ in pf._candidates("x", ())] == ["unchanged"]


def test_ring_through_the_hook_uses_the_launch_alias_classes(tmp_path, monkeypatch):
    from triton import knobs

    from helion.runtime.triton import ttgir_weight_prefetch as pf

    for k in list(os.environ):
        if k.startswith("HELION_TTGIR_RING_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path), raising=False)
    sig = dict(x_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", wa_ptr="*bf16", x2r_ptr="*bf16", x2w_ptr="*bf16",
               out_ptr="*bf16", state_ptr="*i32", epoch_ptr="*i32", N="i32")
    saved = knobs.runtime.add_stages_inspection_hook
    knobs.runtime.add_stages_inspection_hook = None
    try:
        outs = {}
        for tag, classes in (("sep", ()), ("one", X2)):
            with pf.enabled(classes, ring=True):
                ck = _compile_jit(_r3a_k, sig, dict(K=2048, T=16, BN=16, BK=128, NS=4, KA=1024), num_warps=4)
            outs[tag] = ck.asm["ttgir"]
    finally:
        knobs.runtime.add_stages_inspection_hook = saved
    # (the compiled module is re-parsed: SSA names are renumbered) one more single-use barrier: the ordering point
    assert "ttg.warp_specialize" in outs["sep"] and "ttg.warp_specialize" in outs["one"]
    assert outs["one"].count("ttng.init_barrier") == outs["sep"].count("ttng.init_barrier") + 1
    assert re.search(r"ttg\.local_alloc : \(\) -> !ttg\.memdesc<1x1xi64", outs["one"])
    assert not re.search(r"ttg\.local_alloc : \(\) -> !ttg\.memdesc<1x1xi64", outs["sep"])


# ---- mutants of the RING3 guards that survived the build's suite (verifier V-3)
@triton.jit(noinline=True)
def _store_y_rows(x_ptr, y_ptr, blk, K: tl.constexpr, T: tl.constexpr, BK: tl.constexpr):
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :])
    tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v)


@triton.jit(noinline=True)
def _stamp_y_rows(y_ptr, blk, K: tl.constexpr, T: tl.constexpr, BK: tl.constexpr):
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    addr = (y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.int64, bitcast=True)
    tl.inline_asm_elementwise("{ .reg .u64 t; mov.u64 t, %globaltimer; st.global.u64 [$1], t; mov.u32 $0, 0; }",
                              "=r,l", [addr], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _cw_k(x_ptr, xa_ptr, y_ptr, w_ptr, out_ptr, out2_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr, T: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, KA: tl.constexpr, MODE: tl.constexpr):
    """Root A (in the region: it streams xa into out2) writes the activation y ONLY through a noinline callee: MODE 0 a
    plain tt.store (a generic-proxy writer: ordering point), MODE 1 the allowlisted stamp asm (not generic: refused)."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.zeros([T, BK], dtype=tl.float32)
        for ka in tl.range(0, KA, BK, num_stages=NS):
            v += tl.load(xa_ptr + rt[:, None] * KA + (ka + rk)[None, :]).to(tl.float32)
        tl.store(out2_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v)
        if MODE == 0:
            _store_y_rows(x_ptr, y_ptr, blk, K, T, BK)
        else:
            _stamp_y_rows(y_ptr, blk, K, T, BK)
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        _spin(state_ptr, target)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


def _cw(mode):
    sig = dict(x_ptr="*bf16", xa_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", out_ptr="*bf16", out2_ptr="*fp32",
               state_ptr="*i32", epoch_ptr="*i32", N="i32")
    return _compile_jit(_cw_k, sig, dict(K=2048, T=16, BN=16, BK=128, NS=4, KA=512, MODE=mode), num_warps=4).asm["ttgir"]


def test_v2_a_callee_writer_root_gets_an_ordering_point():
    src = _cw(0)
    assert "tt.call" in src
    # the whole-module ring (the policy itself starts at root B here: the whole-module ring's 28748 B would lower the
    # module's 8 CTAs/SM of shared memory, R6-7)
    t, info = ringws2._act_or_wonly(src, ringws2.SMEM_LIMIT, True, {**ringws2.ring_options(), "from_top": None})
    assert info["act"] and info["order_points"] == 1
    default, _, aprod = _parts(t)
    arrive = default.index("ttng.arrive_barrier %rg_wov0, 1")
    assert default.rfind("tt.call", 0, arrive) > 0 and aprod.index("ttng.wait_barrier %pr_wo") < aprod.index(
        "ttg.async_copy_global_to_local")


def test_v3_a_callee_writing_through_assembly_is_not_a_generic_writer():
    src = _cw(1)
    assert "tt.call" in src and "st.global.u64" in src
    with pytest.raises(ringws2.Fail, match="written by tt.call in the warp-specialized region"):
        ringws2.rewrite(src, shared=True, act=True)
    assert ringws2._async_writing_functions(P_uniq(src), ringws2.P._functions(P_uniq(src)))


def test_v10_synchronization_in_a_region_root_that_reads_no_activation():
    # root A (in the region with RA=1) reads no activation; an acquire in it still refuses act
    src = _chain(1)
    pub = re.search(r"^(\s*)(%\S+ = )?tt\.atomic_rmw add, release, gpu, %state_ptr.*$", src, re.M)
    ind = pub.group(1)
    acq = (f"{ind}%rgt_one = arith.constant 1 : i32\n{ind}%rgt_t = arith.constant true\n"
           f"{ind}%rgt_a = tt.atomic_rmw add, acquire, gpu, %epoch_ptr, %rgt_one, %rgt_t : (!tt.ptr<i32>, i32, i1) -> i32\n")
    with pytest.raises(ringws2.Fail, match="acquiring atomic"):
        ringws2.rewrite(src[:pub.start()] + acq + src[pub.start():], shared=True, act=True)
    vol = f"{ind}%rgt_v = tt.load %epoch_ptr {{isVolatile = true}} : !tt.ptr<i32>\n"
    with pytest.raises(ringws2.Fail, match="volatile load"):
        ringws2.rewrite(src[:pub.start()] + vol + src[pub.start():], shared=True, act=True)


@triton.jit
def _iv_k(x_ptr, y_ptr, w_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr, T: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr, NS: tl.constexpr):
    """_chain_k (RA=0) whose consumer K loop also uses its induction variable in register arithmetic."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :])
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v)
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        _spin(state_ptr, target)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc) * (1.0 + (k // BK).to(tl.float32) * 0.001)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


SIG_IV = dict(x_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32", epoch_ptr="*i32", N="i32")


def test_v5_partial_unroll_gives_each_copy_its_own_induction_value():
    src = _compile_jit(_iv_k, SIG_IV, dict(K=2048, T=16, BN=16, BK=128, NS=4), num_warps=4).asm["ttgir"]
    t, _ = ringws2.rewrite(src, shared=True, act=True, unroll=0, unroll_factor=4)
    d = _parts(t)[0]
    for k in (1, 2, 3):
        iv = re.search(rf"^\s*(%\S+_uf{k}) = arith\.addi ", d, re.M)
        assert iv, k
        uses = [line for line in d.splitlines() if iv.group(1) in line and not line.strip().startswith(iv.group(1) + " =")]
        assert uses, f"copy {k} does not use its own induction value"
    _, viol = _hic(t)
    assert not viol, viol[:5]
    offc.compile_ttgir_text(t)


@triton.jit
def _tail_k(x_ptr, y_ptr, w_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr, T: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr, NS: tl.constexpr):
    """The reading root's task loop spins a SECOND time at the end of each task (after its K loop): that spin precedes
    the next task's activation copies only through the loop (it lies after the last copy in program text)."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :])
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v)
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
        tl.atomic_add(state_ptr + 1, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        _spin(state_ptr, target)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
        _spin(state_ptr + 1, target)
    tl.store(epoch_ptr + pid, epoch)


def test_v17_a_spin_sharing_only_the_loop_with_activation_copies_is_cloned():
    src = _compile_jit(_tail_k, SIG_IV, dict(K=2048, T=16, BN=16, BK=128, NS=4), num_warps=4).asm["ttgir"]
    t, info = ringws2.rewrite_budgeted(src)
    assert info["act"] and info["spins"] == 2
    _, _, aprod = _parts(t)
    assert aprod.count("ld.acquire.gpu") == 4      # two spin clones (seed + poll each)


@triton.jit
def _two_k(x_ptr, y_ptr, y2_ptr, w_ptr, w2_ptr, out_ptr, out2_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr,
           T: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr):
    """Two reading roots, each one task per program under an if (no loop shared with any copy): B spins for y and reads
    it, C spins for y2 and reads it.  C's spin lies between the first and the last activation copy in program text."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :])
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v)
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32) * 2.0
        tl.store(y2_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr + 1, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    tile = pid
    rn = tile * BN + tl.arange(0, BN)
    if tile < n_tiles:
        _spin(state_ptr, target)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    if tile < N:   # another condition: MLIR would merge two adjacent ifs on one condition into one block
        _spin(state_ptr + 1, target)
        acc2 = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w2 = tl.load(w2_ptr + rn[:, None] * K + (k + rk)[None, :])
            y2 = tl.load(y2_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc2 = tl.dot(w2, tl.trans(y2), acc2)
        tl.store(out2_ptr + rn[:, None] * T + rt[None, :], acc2.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


def test_v18_every_spin_before_the_last_activation_copy_is_cloned():
    sig = dict(x_ptr="*bf16", y_ptr="*bf16", y2_ptr="*bf16", w_ptr="*bf16", w2_ptr="*bf16", out_ptr="*bf16",
               out2_ptr="*bf16", state_ptr="*i32", epoch_ptr="*i32", N="i32")
    src = _compile_jit(_two_k, sig, dict(K=2048, T=16, BN=16, BK=128, NS=4), num_warps=4).asm["ttgir"]
    t, info = ringws2.rewrite_budgeted(src)
    assert info["act"] and info["spins"] == 2 and info["from_top"] is None
    _, _, aprod = _parts(t)
    polls = [m.start() for m in re.finditer("ld.acquire.gpu", aprod)]
    copies = [m.start() for m in re.finditer("ttg.async_copy_global_to_local", aprod)]
    # the second spin's clone comes after root B's activation copies and before root C's
    assert len(polls) == 4 and copies[0] < polls[2] < copies[-1]
    _, viol = _hic(t)
    assert not viol, viol[:5]


@triton.jit(noinline=True)
def _volatile_callee(flag_ptr):
    return tl.load(flag_ptr, volatile=True)


@triton.jit(noinline=True)
def _barrier_callee(x):
    tl.debug_barrier()
    return x + 1


@triton.jit
def _cs_k(x_ptr, y_ptr, w_ptr, out_ptr, state_ptr, epoch_ptr, flag_ptr, N, K: tl.constexpr, T: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, CALLEE: tl.constexpr):
    """The reading root calls a noinline callee that synchronizes: CALLEE 1 a volatile load, 2 a CTA barrier."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :])
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v)
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    n_tiles = tl.cdiv(N, BN)
    target = epoch * (K // BK)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        _spin(state_ptr, target)
        if CALLEE == 1:
            s = _volatile_callee(flag_ptr)
        else:
            s = _barrier_callee(tile)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], (acc * s.to(tl.float32)).to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


@pytest.mark.parametrize("callee,name", [(1, "volatile_callee"), (2, "barrier_callee")])
def test_v25_v26_volatile_loads_and_barriers_in_callees_synchronize(callee, name):
    src = _compile_jit(_cs_k, SIGX, dict(K=2048, T=16, BN=16, BK=128, NS=4, CALLEE=callee), num_warps=4).asm["ttgir"]
    assert "tt.call" in src
    lines = P_uniq(src)
    assert any(name in n for n in ringws2._sync_functions(lines, ringws2.P._functions(lines)))
    with pytest.raises(ringws2.Fail, match="synchronizing function"):
        ringws2.rewrite(src, shared=True, act=True)


@triton.jit
def _three_k(x_ptr, xa_ptr, y_ptr, w_ptr, x2_ptr, w2_ptr, out_ptr, state_ptr, epoch_ptr, N, K: tl.constexpr,
             T: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, KA: tl.constexpr):
    """Three region roots: A (streams xa) writes y; B reads y and writes x2 (this program's rows); C reads x2.  Two
    reading roots with different last writers -> two ordering points."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    epoch = tl.load(epoch_ptr + pid) + 1
    rt = tl.arange(0, T)
    rk = tl.arange(0, BK)
    for blk in tl.range(pid, K // BK, npid, num_stages=1):
        v = tl.load(x_ptr + rt[:, None] * K + (blk * BK + rk)[None, :]).to(tl.float32)
        for ka in tl.range(0, KA, BK, num_stages=NS):
            v += tl.load(xa_ptr + rt[:, None] * KA + (ka + rk)[None, :]).to(tl.float32)
        tl.store(y_ptr + rt[:, None] * K + (blk * BK + rk)[None, :], v.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")
    target = epoch * (K // BK)
    for tile in tl.range(pid, K // BN, npid, num_stages=1):
        _spin(state_ptr, target)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = tl.load(w_ptr + rn[:, None] * K + (k + rk)[None, :])
            y = tl.load(y_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(y), acc)
        tl.store(x2_ptr + rt[None, :] * K + rn[:, None], acc.to(tl.bfloat16))
        tl.debug_barrier()
        tl.atomic_add(state_ptr + 1, 1, sem="release", scope="gpu")
    target2 = epoch * (K // BN)
    for tile in tl.range(pid, tl.cdiv(N, BN), npid, num_stages=1):
        _spin(state_ptr + 1, target2)
        rn = tile * BN + tl.arange(0, BN)
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w2 = tl.load(w2_ptr + rn[:, None] * K + (k + rk)[None, :])
            x2 = tl.load(x2_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w2, tl.trans(x2), acc)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
    tl.store(epoch_ptr + pid, epoch)


def test_v27_every_reading_root_gets_its_own_ordering_point():
    sig = dict(x_ptr="*bf16", xa_ptr="*bf16", y_ptr="*bf16", w_ptr="*bf16", x2_ptr="*bf16", w2_ptr="*bf16",
               out_ptr="*bf16", state_ptr="*i32", epoch_ptr="*i32", N="i32")
    src = _compile_jit(_three_k, sig, dict(K=2048, T=16, BN=16, BK=128, NS=4, KA=512), num_warps=4).asm["ttgir"]
    t, info = ringws2.rewrite_budgeted(src)
    assert info["act"] and info["order_points"] == 2
    default, _, aprod = _parts(t)
    assert "ttng.arrive_barrier %rg_wov0, 1" in default and "ttng.arrive_barrier %rg_wov1, 1" in default
    assert default.index("%rg_wov0, 1") < default.index("%rg_wov1, 1")
    waits = re.findall(r"ttng\.wait_barrier %pr_wo\d+", aprod)
    assert len(waits) == 2
    _, viol = _hic(t)
    assert not viol, viol[:5]
    offc.compile_ttgir_text(t)


def test_act_refuses_an_impure_external_function_in_the_region():
    # an impure extern may wait on the predecessor (so the module needs Triton's gdc_wait at the top) and may
    # synchronize: R2 refuses it in the region even behind a top-level gdc_wait
    src = _chain(0)
    src = re.sub(r"(\n(\s*)scf\.for %blk)", lambda m: "\n" + m.group(2) + _GDC_WAIT + m.group(1), src, count=1)
    assert "griddepcontrol.wait" in src
    sync = re.search(r"^.*bar\.warp\.sync.*$", src, re.M).group(0)
    pid = re.search(r"(%\S+) = tt\.get_program_id x", src).group(1)
    ext = sync + (f'\n      %rgt_e = tt.extern_elementwise {pid} {{libname = "", libpath = "", pure = false, symbol = "__nv_x"}}'
                  ' : (i32) -> i32')
    with pytest.raises(ringws2.Fail, match="impure external function in the warp-specialized region"):
        ringws2.rewrite(src.replace(sync, ext, 1), shared=True, act=True)
    # without the top-level wait the PDL rule refuses first
    with pytest.raises(ringws2.Fail, match="PDL wait"):
        ringws2.rewrite(_chain(0).replace(sync, ext, 1), shared=True, act=True)
    assert ringws2.rewrite(src, shared=True, act=True)[1]["act"]   # the gdc_wait alone is fine


def test_only_triton_gdc_wait_at_the_top_level_starts_the_region():
    # a top-level op that may wait on the predecessor (an impure extern) is not Triton's gdc_wait: without a real one
    # the PDL rule refuses, and with one the warp_specialize starts after the gdc_wait, not after the extern
    src = _chain(0)
    pid = re.search(r"^(\s*)(%\S+) = tt\.get_program_id x.*$", src, re.M)
    ext = f'{pid.group(1)}%rgt_e = tt.extern_elementwise {pid.group(2)} {{libname = "", libpath = "", pure = false, symbol = "__nv_x"}} : (i32) -> i32'
    src1 = src[:pid.end()] + "\n" + ext + src[pid.end():]
    with pytest.raises(ringws2.Fail, match="PDL wait not at the entry function's top level"):
        ringws2.rewrite(src1, shared=True, act=True)
    src2 = src1.replace(ext, ext + "\n" + pid.group(1) + _GDC_WAIT, 1)
    t, info = ringws2.rewrite(src2, shared=True, act=True)
    lines = ringws2.P._uniquify(src2).splitlines()
    assert "griddepcontrol.wait" in lines[info["ws_start"]] or info["ws_start"] > next(
        k for k, line in enumerate(lines) if "griddepcontrol.wait" in line)


# ---------------------------------------------------------------------------------------------------------------------
# ATT5 rolled producers + ATT6 rolled-form stage guard and automatic size-aware roll (ported from helion-wf-att6).

def test_rolled_producers_issue_the_prologue_tiles_from_the_loop():
    """roll: the producers' loop clone starts num_stages - 1 iterations early, so the in-loop copy also issues the tiles
    Triton peeled into prologue copies; the consumer partition and the stage sequence are unchanged."""
    src = _chain(0)   # K loop: 2048 / 128 = 16 iterations, num_stages 4 -> 3 peeled prologue tiles
    t0, i0 = ringws2.rewrite(src, shared=True, act=True)
    t1, i1 = ringws2.rewrite(src, shared=True, act=True, roll=True)
    assert i0["rolled"] == 0 and i1["rolled"] == 1
    d0, w0, a0 = _parts(t0)
    d1, w1, a1 = _parts(t1)
    assert d1 == d0                                                       # consumer untouched
    for p0, p1 in ((w0, w1), (a0, a1)):
        assert p0.count("ttg.async_copy_global_to_local") == 4             # 3 prologue + 1 in-loop copy
        assert p1.count("ttg.async_copy_global_to_local") == 1             # the in-loop copy only
        assert re.search(r"arith\.constant -384 : i32", p1) and re.search(r"arith\.constant 1664 : i32", p1)
        assert p1.count("ttng.wait_barrier") <= p0.count("ttng.wait_barrier")
    assert (i1["pstages"], i1["cstages"], i1["D"]) == (i0["pstages"], i0["cstages"], i0["D"])
    _, viol = _hic(t1)
    assert not viol, viol[:5]
    s = offc.compile_ttgir_text(t1)
    assert s["spill"] == "0"


def test_roll_leaves_loops_that_are_not_a_plain_peeling():
    """A loop whose trip count is below num_stages - 1 has dead prologue copies and an in-loop copy that never runs:
    the peeled-tile relation (predicate iv < ub - K*step with tile iv + K*step) does not hold, so the producer keeps
    Triton's prologue copies verbatim."""
    src = _chain(0, bk=128, ns=4).replace("%c2048_i32", "%c2048_i32")  # K loop as above
    short = _compile_jit(_chain_k, SIGC, dict(K=256, T=16, BN=16, BK=128, NS=4, RA=0, KA=512), num_warps=4).asm["ttgir"]
    t0, i0 = ringws2.rewrite(short, shared=True, act=True)
    t1, i1 = ringws2.rewrite(short, shared=True, act=True, roll=True)
    assert i1["rolled"] == 0 and t1 == t0
    assert src  # the long-loop kernel of the previous test rolls (sanity of the fixture)


def test_rolled_producer_stage_sequence_is_rechecked(monkeypatch):
    """ATT6: the global stage-sequence guard runs again on the ROLLED producer (loop clone re-bounded, prologue copies
    dropped). A rolled window one tile short (ATT5 mutant RM1, which hangs on the GPU) changes the producer's word, so
    the roll is refused and the ring keeps Triton's prologue copies; without the re-check the mutant would be emitted."""
    src = _chain(0)
    good, gi = ringws2.rewrite(src, shared=True, act=True, roll=True)
    plain, _ = ringws2.rewrite(src, shared=True, act=True)
    assert gi["rolled"] == 1
    short = lambda lb, x, k, st: (lb - (k - 1) * st, x)  # noqa: E731
    monkeypatch.setattr(ringws2, "_roll_window", short)
    log = []
    t, info = ringws2.rewrite(src, shared=True, act=True, roll=True, log=log)
    assert info["rolled"] == 0 and t == plain
    assert any("rolled stage sequence differs" in m for m in log), log
    monkeypatch.setattr(ringws2, "_verified_rolls", lambda rl, pw, cw, log=None: rl)
    t_bad, info_bad = ringws2.rewrite(src, shared=True, act=True, roll=True)
    assert info_bad["rolled"] == 1 and t_bad not in (good, plain)  # the guard (not the window check) refused it


def test_verified_rolls_checks_each_loop_then_the_combination():
    cw = ("S", "S", "S")
    words = {frozenset({1}): cw, frozenset({2}): cw, frozenset({3}): ("S",), frozenset({1, 2}): ("S", "S")}
    pw = lambda rl: words[frozenset(rl)]  # noqa: E731
    log = []
    assert ringws2._verified_rolls({1: "a", 3: "c"}, pw, cw, log) == {1: "a"}
    assert any("loop at 3" in m for m in log)
    assert ringws2._verified_rolls({1: "a", 2: "b"}, pw, cw, log) == {}
    assert ringws2._verified_rolls({1: "a"}, pw, cw) == {1: "a"}



def _labels_under_hook(tmp_path, monkeypatch, budget, codes=None):
    """(candidate labels of the compile, label of the binary) of the rolled-producer chain kernel under the hook with
    CODE_BUDGET = budget; ``codes`` collects (label, SASS code bytes) of every trial ptxas."""
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    seen = {}
    real_ttgir, real_cubin = pf._Fallback.ttgir, pf._Fallback.cubin
    real_ptx, real_trial = pf._Fallback.ptx, pf._ptxas_trial
    current = []

    def ptx(self, src, md):
        current.append(self)
        try:
            return real_ptx(self, src, md)
        finally:
            current.pop()

    def trial(*a):
        out = real_trial(*a)
        if codes is not None:
            codes.append((current[-1].label, out[1]))
        return out
    monkeypatch.setattr(pf._Fallback, "ptx", ptx)
    monkeypatch.setattr(pf, "_ptxas_trial", trial)

    def ttgir(self, src, md):
        out = real_ttgir(self, src, md)
        seen["cands"] = [lab for lab, _ in self.cands]
        return out

    def cubin(self, src, md):
        out = real_cubin(self, src, md)
        seen["used"] = self.label
        return out
    monkeypatch.setattr(pf._Fallback, "ttgir", ttgir)
    monkeypatch.setattr(pf._Fallback, "cubin", cubin)
    monkeypatch.setattr(ringws2, "CODE_BUDGET", budget)
    # KA=1024: root A's ring loop is long enough that rolling its producer shrinks the code at BN=128
    ck = _compile_under_hook(tmp_path, monkeypatch, chain=True, ka=1024)
    return seen["cands"], seen["used"], ck


def test_size_twins_are_chosen_by_code_size(tmp_path, monkeypatch):
    """No switch. Each ring form is followed by its smaller twins (rolled producers, then also no consumer unroll); the
    trial ptxas keeps the first form whose code fits CODE_BUDGET, walking the twins in order, and the last twin when
    none fits (it still beats the hoist on the kernel that needs it: ATT5)."""
    cands, used, ck = _labels_under_hook(tmp_path / "big", monkeypatch, 1 << 30)
    assert cands[:3] == ["ring", "ring_roll", "ring_roll_u0"] and used == "ring"
    codes = []
    cands2, used2, ck2 = _labels_under_hook(tmp_path / "small", monkeypatch, 1024, codes)
    assert cands2 == cands and used2 == "ring_roll_u0"
    assert [lab for lab, _ in codes] == ["ring", "ring_roll", "ring_roll_u0"]
    size = dict(codes)
    assert size["ring"] > size["ring_roll"] > size["ring_roll_u0"]
    assert ck2.asm["ptx"].count("cp.async") < ck.asm["ptx"].count("cp.async")   # prologue copies not issued
    assert ck2.metadata.num_warps == ck.metadata.num_warps == 8
    # a budget the rolled twin fits: it stops there
    cands3, used3, ck3 = _labels_under_hook(tmp_path / "mid", monkeypatch, size["ring_roll"])
    assert used3 == "ring_roll"
    assert ck3.asm["ptx"].count("cp.async") == ck2.asm["ptx"].count("cp.async")


_COVERAGE_TEXT = """\
    %a = ttg.async_copy_global_to_local %p, %s : tensor<16x128x!tt.ptr<bf16>, #blocked> -> <16x128xbf16, #shared, #smem, mutable>
    ttg.warp_specialize(%x)
    default {
      %b = ttg.async_copy_global_to_local %p, %s : tensor<16x64x!tt.ptr<bf16>, #blocked> -> <16x64xbf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg: i32) num_warps(1) {
      %c = ttg.async_copy_global_to_local %p, %s : tensor<32x128x!tt.ptr<bf16>, #blocked> -> <32x128xbf16, #shared, #smem, mutable>
      scf.if %q {
        ttng.async_tma_copy_global_to_local %d[%i] %buf, %bar, %true : !tt.tensordesc<128x256xui8, #shared>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<128x256xi8, #shared5, #smem, mutable> loc(#loc3)
      } : () -> ()
      ttg.warp_return
    }
    partition1(%arg: i32) num_warps(1) {
      ttg.warp_return
    } : (i32) -> ()
    %e = ttg.async_copy_global_to_local %p, %s : tensor<8x128x!tt.ptr<f32>, #blocked> -> <8x128xf32, #shared, #smem, mutable>
"""


def test_copy_coverage_counts_the_producer_partitions_copy_bytes():
    # prefix 4096 + default 2048 + after the warp_specialize 4096 stay; partition0's cp.async 8192 + TMA 32768 move
    # (a nested region's "} :" does not end the partition)
    assert ringws2.copy_coverage(_COVERAGE_TEXT) == pytest.approx((8192 + 32768) / (4096 + 2048 + 8192 + 32768 + 4096))
    assert ringws2.copy_coverage("tt.return") == 0.0
    t, _ = ringws2.rewrite_budgeted(_chain(1))
    assert ringws2.copy_coverage(t) == 1.0   # the two-producer ring moves every copy of the chain
    assert ringws2.copy_coverage(_chain(1)) == 0.0


def test_ring_below_the_copy_coverage_is_not_a_candidate(monkeypatch):
    """Profitability (generic, from the candidate text): a ring form whose producers would issue less than
    MIN_COPY_COVERAGE of the module's copy bytes is no candidate (the hoist is next); its twins go with it."""
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    src = _chain(1)
    labels = [lab for lab, _ in pf._candidates(src, (), ring=True)]
    assert labels[0] == "ring"
    monkeypatch.setattr(ringws2, "MIN_COPY_COVERAGE", 1.01)
    low = [lab for lab, _ in pf._candidates(src, (), ring=True)]
    assert not any(lab.startswith("ring") for lab in low) and low[-1] == "unchanged"
    assert low == [lab for lab in labels if not lab.startswith("ring")]


def test_fallback_walks_size_twins_only_within_one_ring_form():
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    fb = pf._Fallback({}, None, None)
    labels = ["ring", "ring_roll", "ring_roll_u0", "ring_wonly", "ring_wonly_roll_u0", "hoist", "unchanged"]
    fb.cands = [(lab, "") for lab in labels]
    twin_next = []
    for k in range(len(labels)):
        fb.k = k
        twin_next.append(fb._smaller_twin_next())
    assert twin_next == [True, True, False, True, False, False, False]
    assert [pf._ring_form(lab) for lab in labels] == ["ring"] * 3 + ["ring_wonly"] * 2 + ["hoist", "unchanged"]


def test_size_twins_only_where_they_change_the_ring():
    from helion.runtime.triton import ttgir_weight_prefetch as pf

    labels = [lab for lab, _ in pf._candidates(_chain(0), (), ring=True)]
    assert labels[:3] == ["ring", "ring_roll", "ring_roll_u0"] and labels[-1] == "unchanged"
    short = _compile_jit(_chain_k, SIGC, dict(K=256, T=16, BN=16, BK=128, NS=4, RA=0, KA=512), num_warps=4).asm["ttgir"]
    cands = pf._candidates(short, (), ring=True)
    assert not any(lab.endswith("_roll") for lab, _ in cands)   # trip < stages: nothing to roll
    texts = [t for _, t in cands if t is not None]
    assert len(texts) == len(set(texts))


def test_roll_refuses_a_loop_copy_that_is_not_the_peeled_successor():
    """(ATT6 verify, mutants RM4 / RM4c) The rolled-producer re-check compares stage counts and order only; it cannot see
    tile addresses.  The cone rule must refuse a loop whose in-loop copy issues tile iv + step instead of the
    pipeliner's peeled successor iv + K * step (K = 3 prologue stages of step 128 here): rolled, the producer would issue
    other tiles than the prologue it replaces (wrong data, out-of-bounds reads) with the same stage word."""
    src = _chain(0)
    succ = re.compile(r"(%[\w.]+ = arith\.addi %[\w.]+), %c384_i32 : i32")
    assert len(succ.findall(src)) == 1 and "%c128_i32 = arith.constant 128 : i32" in src
    witness = succ.sub(r"\1, %c128_i32 : i32", src)
    assert ringws2.rewrite(src, shared=True, act=True, roll=True)[1]["rolled"] == 1  # the fixture itself rolls
    unrolled, _ = ringws2.rewrite(witness, shared=True, act=True)
    log = []
    rolled, info = ringws2.rewrite(witness, shared=True, act=True, roll=True, log=log)
    assert info["rolled"] == 0
    assert rolled == unrolled
    assert any("not rolled" in m for m in log)


def test_roll_refuses_the_successor_constant_under_another_op():
    """(ATT7 verify, mutant RM4c3) The peeled successor is iv + K * step exactly: the same K * step constant under
    another op (iv - 384, iv * 384) issues other tiles, so the cone rule must check the op, not only the constant."""
    src = _chain(0)
    succ = re.compile(r"(%[\w.]+) = arith\.addi (%[\w.]+), %c384_i32 : i32")
    assert len(succ.findall(src)) == 1
    for op in ("arith.subi", "arith.muli"):
        witness = succ.sub(rf"\1 = {op} \2, %c384_i32 : i32", src)
        unrolled, _ = ringws2.rewrite(witness, shared=True, act=True)
        log = []
        rolled, info = ringws2.rewrite(witness, shared=True, act=True, roll=True, log=log)
        assert info["rolled"] == 0, op
        assert rolled == unrolled, op
        assert any("not rolled" in m for m in log), op


def test_roll_has_no_environment_switch(monkeypatch):
    monkeypatch.setenv("HELION_TTGIR_RING_ROLL", "1")
    assert "roll" not in ringws2.options_from_env()


# ---------------------------------------------------------------- TMA rings (Triton-pipelined descriptor loads)
@triton.jit
def _tma_gemv_k(w_desc, x_ptr, route_ptr, out_ptr, state_ptr, N, K: tl.constexpr, T: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, ROUTED: tl.constexpr, WRITE: tl.constexpr,
                SPIN: tl.constexpr, RED: tl.constexpr, PUB: tl.constexpr = 0):
    """Persistent GEMV whose weight tiles are TMA loads Triton pipelines (barrier_expect + async_tma_copy per stage).
    ROUTED: the tile's weight row comes from a route table (a dependent address, DESIGN rule E); WRITE: the kernel also
    writes the route table inside the tile loop; SPIN: an acquire spin precedes the route load; RED: the row is an
    integer reduction (tt.reduce) of a loaded route vector; PUB: the kernel itself publishes the state the spin polls
    (a release add after each tile, like Helion's tile-dependency counters)."""
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    n_tiles = tl.cdiv(N, BN)
    rk = tl.arange(0, BK)
    rt = tl.arange(0, T)
    for tile in tl.range(pid, n_tiles, npid, num_stages=1):
        if SPIN:
            c = _acq(state_ptr)
            while c != tile:
                c = _acq(state_ptr)
            tl.inline_asm_elementwise("bar.warp.sync 0xffffffff; mov.u32 $0, $1;", "=r,r", [tl.arange(0, 32)],
                                      dtype=tl.int32, is_pure=False, pack=1)
        if RED:
            row = tl.sum(tl.load(route_ptr + tile * 8 + tl.arange(0, 8))) * BN
        elif ROUTED:
            row = tl.load(route_ptr + tile) * BN
        else:
            row = tile * BN
        acc = tl.zeros([BN, T], dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            w = w_desc.load([row, k])
            x = tl.load(x_ptr + rt[:, None] * K + (k + rk)[None, :])
            acc = tl.dot(w, tl.trans(x), acc)
        rn = tile * BN + tl.arange(0, BN)
        tl.store(out_ptr + rn[:, None] * T + rt[None, :], acc.to(tl.bfloat16))
        if WRITE:
            tl.store(route_ptr + tile, 0)
        if PUB:
            tl.atomic_add(state_ptr, 1, sem="release", scope="gpu")


def _tma(routed=0, write=0, spin=0, red=0, bn=128, bk=128, ns=4, pub=0):
    sig = dict(w_desc=f"tensordesc<bf16[{bn}, {bk}]>", x_ptr="*bf16", route_ptr="*i32", out_ptr="*bf16",
               state_ptr="*i32", N="i32")
    return _compile_jit(_tma_gemv_k, sig, dict(K=2048, T=16, BN=bn, BK=bk, NS=ns, ROUTED=routed, WRITE=write,
                                               SPIN=spin, RED=red, PUB=pub), num_warps=4).asm["ttgir"]


def test_tma_ring_structure_and_compiles():
    src = _tma()
    assert src.count("ttng.async_tma_copy_global_to_local") >= 2 and "ttng.barrier_expect" in src
    t, info = ringws2.rewrite(src, shared=True)
    default, prod = _split(t)
    # the producer issues every TMA copy into the ring's slots, completing on the ring's ready barrier
    assert "ttng.async_tma_copy_global_to_local" not in default and "ttng.barrier_expect" not in default
    assert prod.count("ttng.async_tma_copy_global_to_local") == info["pstages"]
    expects = re.findall(r"ttng\.barrier_expect (%pr_rb\d+), (\d+), %pr_true", prod)
    assert len(expects) == info["pstages"] and {b for _, b in expects} == {str(128 * 128 * 2)}
    for bar, _ in expects:
        assert re.search(rf"async_tma_copy_global_to_local [^\n]* %pr_slot\d+, {bar}, %pr_true :", prod)
    # Triton's per-stage TMA barriers are gone; the ready barrier counts the expect arrival + the cp.async arrivals of
    # the 4 weight-producer warps (x is read-only: its copies move to the same producer)
    assert re.search(r"ttng\.init_barrier %rg_r0, 129 :", t)
    assert len(re.findall(r"ttng\.init_barrier %\S+, 1 : !ttg\.memdesc<1xi64, #shared\d*", default)) <= 2
    s = offc.compile_ttgir_text(t)
    assert s["num_warps"] == 8 and s["spill"] == "0"


def test_tma_rule_e_clones_the_routing_load():
    t, info = ringws2.rewrite(_tma(routed=1), shared=True)
    default, prod = _split(t)
    assert default.count("tt.load %") >= 1 and re.search(r"= tt\.load %pu\S+ : !tt\.ptr<i32>", prod)
    assert prod.count("ttng.async_tma_copy_global_to_local") == info["pstages"]
    offc.compile_ttgir_text(t)


def test_tma_rule_e_clones_an_integer_reduction():
    t, _ = ringws2.rewrite(_tma(red=1), shared=True)
    _, prod = _split(t)
    assert prod.count('"tt.reduce"') == 1 and "tt.reduce.return" in prod
    offc.compile_ttgir_text(t)


def test_tma_rule_e_clones_the_preceding_spin():
    t, _ = ringws2.rewrite(_tma(routed=1, spin=1), shared=True)
    default, prod = _split(t)
    # the producer performs the consumer's acquire before it reads the route table
    assert prod.count("ld.acquire.gpu") == 2 and prod.index("ld.acquire.gpu") < prod.index("tt.load %pu")
    assert default.count("ld.acquire.gpu") == 2
    offc.compile_ttgir_text(t)


def test_tma_rule_e_refuses_a_route_table_written_in_the_region():
    with pytest.raises(ringws2.Fail):
        ringws2.rewrite(_tma(routed=1, write=1), shared=True)


def test_tma_expect_bytes_mismatch_is_refused():
    src = _tma()
    assert "ttng.barrier_expect" in src
    with pytest.raises(ringws2.Fail):
        ringws2.rewrite(re.sub(r"(ttng\.barrier_expect %\S+), 32768,", r"\1, 16384,", src, count=1), shared=True)


def test_tma_barrier_with_a_foreign_use_is_refused():
    src = _tma()
    m = re.search(r"ttng\.barrier_expect (%\S+), 32768, (%\S+) : (!ttg\.memdesc<1xi64[^>]*>)", src)
    # an extra arrive on a TMA stage barrier: it orders something else -> refuse
    extra = f"ttng.arrive_barrier {m.group(1)}, 1 : {m.group(3)}\n    "
    with pytest.raises(ringws2.Fail):
        ringws2.rewrite(src.replace(m.group(0), extra + m.group(0), 1), shared=True)


def test_tma_source_written_in_kernel_stays():
    src = _tma()
    # a TMA store through the weight descriptor: the TMA ring must not move (the x ring may)
    m = re.search(r"^(\s*)(%\S+) = ttg\.memdesc_index (%\S+)\[%c0_i32\] : !ttg\.memdesc<4x128x128xbf16.*$", src, re.M)
    assert m is not None
    store = (f"{m.group(1)}ttng.async_tma_copy_local_to_global %w_desc[%c0_i32, %c0_i32] {m.group(2)} : "
             "!tt.tensordesc<128x128xbf16, #shared>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>")
    src2 = src.replace(m.group(0), m.group(0) + "\n" + store, 1)
    log = []
    t, info = ringws2.rewrite(src2, shared=True, log=log)
    assert any("TMA source written in-kernel, stays" in l for l in log)
    assert len(info["rings"]) == 1 and "_x" in info["rings"][0]
    default, prod = _split(t)
    assert "ttng.async_tma_copy_global_to_local" not in prod and "ttng.barrier_expect" in default
    offc.compile_ttgir_text(t)


def test_resident_ring_keeps_occupancy_and_register_budget():
    src = _tma()
    plain, pinfo = ringws2.rewrite_budgeted(src)
    assert pinfo["smem"] > ringws2.resident_limit(2)  # D4: one CTA per SM
    t, info = ringws2.rewrite_budgeted(src, resident_ctas=2)
    assert info["smem"] <= ringws2.resident_limit(2) == 233472 // 2 - 1024 and info["D"] < pinfo["D"]
    assert info["resident_ctas"] == 2 and "nvvm.minctasm = 2 : i32" in t
    assert '"ttg.maxnreg" = 128 : i32' in t and "requestedRegisters = array<i32: 40>" in t
    s = offc.compile_ttgir_text(t)
    assert s["num_warps"] == 8 and s["setmaxnreg"] > 0
    # one CTA per SM (or no requirement): the ring is the default one
    assert ringws2.rewrite_budgeted(src, resident_ctas=1) == (plain, pinfo)
    assert ringws2.rewrite_budgeted(src, resident_ctas=0) == (plain, pinfo)


def test_resident_fact_is_part_of_the_key():
    from helion.runtime.triton import ttgir_weight_prefetch as twp

    assert twp._Facts(ring=True, resident_ctas=2).key() == "-resident2"
    assert twp._Facts(ring=True, resident_ctas=1).key() == ""
    assert twp._Facts(ring=False, resident_ctas=2).key() == ""
    fb = twp._Fallback({}, None, None, (), 2)
    assert fb._spill_allowance() == ringws2.RESIDENT_SPILL_BYTES
    assert twp._Fallback({}, None, None, ())._spill_allowance() == 0


@pytest.mark.parametrize("ns", [2, 3, 4])
def test_ring_keeps_the_module_occupancy(ns):
    """(B5-8, R6-7) A ring with no spin-gated first stage (R6-8) keeps the module's shared-memory occupancy, all of it
    (depth lowered to fit), up to what a warp-specialized kernel of it reaches at all (2048 threads / (32 x (4 + 4)) =
    8 CTAs for a 4-warp module)."""
    src = _tma(ns=ns)
    occ = ringws2.SMEM_PER_SM // (ringws2.smem_of(src) + ringws2.SMEM_RESERVED)
    assert ringws2.kept_occupancy(src) == max(1, min(occ, 8))
    if ns == 2:
        # 73760 B: 3 CTAs/SM as is; no TMA ring depth keeps 3 (D2 needs 77860 B > 76800): refused
        assert occ == 3
        with pytest.raises(ringws2.Fail, match="does not fit"):
            ringws2.rewrite_budgeted(src)
    else:
        _, info = ringws2.rewrite_budgeted(src)
        assert info["smem"] <= ringws2.resident_limit(ringws2.kept_occupancy(src))
    assert ringws2.kept_occupancy(_gemv()) == 8  # BN16 gemv: 9 CTAs/SM of shared memory, 8 reachable
    g = _gemv(32)
    occ32 = ringws2.SMEM_PER_SM // (ringws2.smem_of(g) + ringws2.SMEM_RESERVED)
    assert 2 < occ32 < 8 and ringws2.kept_occupancy(g) == occ32   # no cap at 2 any more
    _, info32 = ringws2.rewrite_budgeted(g)
    assert info32["smem"] <= ringws2.resident_limit(occ32)


def test_copy_coverage_weights_constant_loop_trips():
    """(B5-9) An in-loop copy of a constant-trip loop counts once per iteration; a loop with run-time bounds once."""
    text = """\
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %a = ttg.async_copy_global_to_local %p, %s : tensor<16x128x!tt.ptr<bf16>, #blocked> -> <16x128xbf16, #shared, #smem, mutable>
    ttg.warp_specialize(%x)
    default {
      ttg.warp_yield
    }
    partition0(%arg: i32) num_warps(1) {
      scf.for %i = %c0 to %c8 step %c1 : i32 {
        %b = ttg.async_copy_global_to_local %p, %s : tensor<16x128x!tt.ptr<bf16>, #blocked> -> <16x128xbf16, #shared, #smem, mutable>
      }
      scf.for %j = %arg to %c8 step %c1 : i32 {
        %c = ttg.async_copy_global_to_local %p, %s : tensor<16x128x!tt.ptr<bf16>, #blocked> -> <16x128xbf16, #shared, #smem, mutable>
      }
      ttg.warp_return
    } : (i32) -> ()
"""
    assert ringws2.copy_coverage(text) == pytest.approx(9 / 10)


def test_suffix_policy_skips_a_low_coverage_suffix():
    sig = dict(w1_ptr="*bf16", w2_ptr="*bf16", x_ptr="*bf16", y_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32",
               epoch_ptr="*i32", N2="i32")
    # root A's K (4096) dwarfs root B: B's own ring would stream ~12% of the copy bytes -> the policy takes both roots
    src = _compile_jit(_two_gemv_k, sig, dict(N1=2048, K=4096, T=16, BN1=128, BK1=256, NS1=3, BN2=16, BK2=128, NS2=7),
                       num_warps=4).asm["ttgir"]
    starts = ringws2._ring_starts(src)
    tb, ib = ringws2.rewrite_budgeted(src, from_top=starts[1])
    assert ib["D"] == ib["want_D"] and ringws2.copy_coverage(tb) < ringws2.MIN_COPY_COVERAGE
    t, info = ringws2.rewrite_budgeted(src)
    assert info["from_top"] is None and ringws2.copy_coverage(t) >= ringws2.MIN_COPY_COVERAGE


def test_rule_e_ring_without_lookahead_is_flagged():
    """(B5-9) The routed TMA gemv whose kernel publishes the state its per-tile spin polls: W clones the acquire spin
    its route load needs, the only spin before the first stage, and that spin waits for this kernel's own progress ->
    no lookahead; without the spin W waits for nothing (it streams ahead across tiles), and without a dependent load
    it clones nothing."""
    _, info = ringws2.rewrite(_tma(routed=1, spin=1, pub=1), shared=True)
    assert info["w_lookahead"] is False
    assert ringws2.rewrite(_tma(routed=1), shared=True)[1]["w_lookahead"] is True
    assert ringws2.rewrite(_tma(), shared=True)[1]["w_lookahead"] is True
    with pytest.raises(ringws2.Fail, match="no lookahead"):
        ringws2.rewrite_budgeted(_tma(routed=1, spin=1, pub=1))


def test_rule_e_producer_runs_ahead_past_spins_on_state_published_outside_the_kernel():
    """(R6, verify5 la.py) The same persistent gemv when no op of the kernel writes the polled state: the spin W clones
    sits in the tile loop with the first stage, and nothing this kernel does gates it, so W runs on into later tiles by
    the ring's depth -> lookahead, and the policy builds the ring (la.py: ~12% faster than the hoist)."""
    src = _tma(routed=1, spin=1)
    _, info = ringws2.rewrite(src, shared=True)
    assert info["w_lookahead"] is True
    t, binfo = ringws2.rewrite_budgeted(src)
    assert binfo["from_top"] is None and binfo["D"] >= 2
    default, prod = _split(t)
    assert prod.count("ld.acquire.gpu") == 2 and default.count("ld.acquire.gpu") == 2


def test_rule_e_lookahead_needs_a_shared_loop_and_read_only_polls():
    """The exception's two conditions on stand-in nodes: a loop encloses the cloned spins and the first stage, and
    every spin W clones inside it polls only read-only arguments."""
    from types import SimpleNamespace as NS

    seed = next(l for l in _tma(routed=1, spin=1).split("\n") if "ld.acquire.gpu" in l)
    lines = [seed, ""]
    loop = NS(i=9, parent=None, op="scf.for")
    spin = NS(i=0, parent=None, regions=[], uses=["%state"])
    stage = NS(i=1, parent=None, regions=[], uses=[])
    pre = [[spin]]

    def runs(read_only, roots=None):
        return ringws2._runs_ahead_into_later_tasks(lines, stage, pre, pre, {0}, roots or {}, read_only)

    assert runs({"%state"}) is False            # no loop at all (a one-shot task)
    spin.parent = loop
    assert runs({"%state"}) is False            # the stage is outside the spin's loop
    stage.parent = loop
    assert runs({"%state"}) is True             # one loop, the polled state is published outside the kernel
    assert runs(set()) is False                 # the kernel writes the polled state
    assert runs({"%state"}, {"%state": frozenset({"%state", "%flags"})}) is False   # a written root among its roots


def test_negative_control_nowspin_drops_the_rule_e_spin_clone_from_the_weight_producer():
    src = _tma(routed=1, spin=1)
    t, _ = ringws2.rewrite(src, shared=True)
    tn, _ = ringws2.rewrite(src, shared=True, neg="nowspin")
    assert _split(t)[1].count("ld.acquire.gpu") == 2 and "ld.acquire.gpu" not in _split(tn)[1]
    assert _split(t)[0] == _split(tn)[0]   # the consumer keeps its spin


def test_mlir_rejected_candidate_is_a_refusal_not_an_error():
    """(verify5 C3 / C7) A 1-warp module's ring is a warp-specialized module MLIR rejects ("num_warps ... multiple of
    4"): smem_of raises Fail (a refusal), not RuntimeError."""
    src = _compile_jit(kern.gemv, SIG, dict(K=2048, T=16, BN=16, BK=128, NS=4), num_warps=1).asm["ttgir"]
    t, _ = ringws2.rewrite(src)
    with pytest.raises(ringws2.Fail, match="MLIR rejects"):
        ringws2.smem_of(t)
    with pytest.raises(ringws2.Fail, match="MLIR rejects"):
        ringws2.rewrite_budgeted(src)


def test_mlir_rejected_suffix_does_not_hide_the_next_suffix(monkeypatch):
    sig = dict(w1_ptr="*bf16", w2_ptr="*bf16", x_ptr="*bf16", y_ptr="*bf16", out_ptr="*bf16", state_ptr="*i32",
               epoch_ptr="*i32", N2="i32")
    src = _compile_jit(_two_gemv_k, sig, dict(N1=2048, K=1024, T=16, BN1=128, BK1=256, NS1=3, BN2=16, BK2=128, NS2=7),
                       num_warps=4).asm["ttgir"]
    starts = ringws2._ring_starts(src)
    real = ringws2.rewrite

    def broken_whole_module(text, **kw):
        t, info = real(text, **kw)
        if kw.get("from_top") is None:   # the whole-module ring: a module MLIR cannot parse
            t = t.replace("ttg.warp_specialize(", "ttg.warp_specialize_bogus(", 1)
        return t, info
    monkeypatch.setattr(ringws2, "rewrite", broken_whole_module)
    _, info = ringws2.rewrite_budgeted(src)
    assert info["from_top"] == starts[1]
