"""Warp-specialized weight ring built from Triton's own pipelined cp.async loops (setting ``ttgir_weight_ring``, on).

Installed by :mod:`.ttgir_weight_prefetch`'s stages hook as the first candidates of its fallback chain (two-producer
ring, weight-only ring, hoist, module unchanged); on any refusal (:class:`Fail`) the hoist runs instead, and a later
stage that fails on a ring (make_llir, ptxas, spills) falls back to the next candidate.

Runs on the TritonGPU IR after make_ttgir of a NON-warp-specialized kernel (what the stages hook sees).  It reuses the
hoist pass's analyses, so it is never less conservative than the hoist: which arguments a kernel writes
(``_written_params``, closed over the launch's storage alias classes), which inline assembly is known (the exact-template
allowlist; any other assembly writes every argument and may wait on the PDL predecessor), the PDL rule (a module that
may wait on its predecessor needs Triton's exact ``gdc_wait`` at the entry's top level, before the warp_specialize,
and no assembly off the allowlist), and no system-scope op or integer-to-pointer cast in any function.

* Rings: ttg.local_alloc <D x tile> buffers filled by ttg.async_copy_global_to_local whose source is a kernel argument
  no op of the kernel writes (the canonical pass's analysis).  All selected rings move to the function entry.
* Slot handoff: per slot a ready mbarrier (count = producer threads; each producer thread's
  cp.async.mbarrier.arrive.noinc fires when its copies land) and an empty mbarrier (count 1; the consumer's
  arrive_barrier after its last local_load of the stage, or a tcgen05 commit after the MMA reading the slot).
* GLOBAL stage counter c (slot c % D, phase (c / D) & 1) carried through every loop / if that contains a stage, in both
  partitions: the producer walks the concatenated copy stream of every root and task (sgl-nvidia-mega mla/pipeline.cuh
  scheme: stage = task & 3 with a global task counter); streaming continues across task and root boundaries.
* Default region = the original function body, minus the moved copies (their commit groups stay, possibly empty, so
  the async_wait counts of the copies that remain - e.g. activations written in-kernel - are unchanged), plus the
  wait / release / counter.  partition0 (same warp count, so every distributed layout is reused verbatim) = the loop /
  if nest restricted to the backward cones of the moved copies (pure index math, program id, loop bounds), with
  Triton's own copy ops: prologue copies verbatim, in-loop copies guarded by the pipeliner's scalar predicate (a stage
  exists iff its copy is live).  Bytes consumed by every MMA are unchanged; only the issuing warps and the slot change.
* shared=True (stage 2): rings of the same role (k-th ring of a stage) in different roots share ONE byte ring
  memdesc<D x Bmax x i8>; each ring is a memdesc_reinterpret of the WHOLE byte ring as (D*Bmax/b) slots of its own
  tile type/encoding, indexed s*Bmax/b (Triton forbids reinterpreting a subview), so smem = D * max stage bytes.

Fail-closed: raises Fail(reason); the hook then leaves the module unchanged.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import os
import tempfile

from . import ttgir_weight_prefetch as P

V = r"%[\w$.#-]+"
_FOR = re.compile(
    rf"^(?P<ind>\s*)(?:(?P<res>{V})(?::(?P<n>\d+))?\s*=\s*)?scf\.for\s+(?P<iv>{V})\s*=\s*(?P<lb>{V})"
    rf"\s+to\s+(?P<ub>{V})\s+step\s+(?P<st>{V})\s*(?:iter_args\((?P<ia>.*?)\)\s*->\s*\((?P<ty>.*?)\))?"
    r"\s*:\s*(?P<ivt>\w+)\s*\{\s*$")
_IF = re.compile(rf"^(?P<ind>\s*)(?:(?P<res>{V})(?::(?P<n>\d+))?\s*=\s*)?scf\.if\s+(?P<c>{V})"
                 r"\s*(?:->\s*\((?P<ty>.*?)\))?\s*\{\s*$")
_COPY_RE = re.compile(
    rf"^\s*(?P<res>{V})\s*=\s*ttg\.async_copy_global_to_local\s+(?P<src>{V}),\s*(?P<dst>{V})"
    rf"(?:\s+mask\s+(?P<mask>{V}))?(?:\s+other\s+(?P<other>{V}))?")
_WHILE = re.compile(rf"^(?P<ind>\s*)(?:(?P<res>{V})(?::(?P<n>\d+))?\s*=\s*)?scf\.while\s*\((?P<args>.*?)\)\s*:\s*"
                    r"\((?P<ty>.*?)\)\s*->\s*(?P<rty>.*?)\s*\{\s*$")
# Triton's pipelined TMA copy: descriptor[coordinates] -> slot view, completion barrier, predicate.
_TMA_RE = re.compile(
    rf"^\s*ttng\.async_tma_copy_global_to_local\s+(?P<desc>{V})\[(?P<coords>[^\]]*)\]\s+(?P<dst>{V}),\s*(?P<bar>{V}),"
    rf"\s*(?P<pred>{V})\s*:\s*(?P<tys>.*)$")
_EXPECT_RE = re.compile(rf"^\s*ttng\.barrier_expect\s+(?P<bar>{V}),\s*(?P<bytes>\d+),\s*(?P<pred>{V})\s*:")
_MMA_OPS = frozenset({"ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled"})
# Views through which a slot reaches its readers unchanged (same bytes, another shape / order).
_SLOT_VIEWS = frozenset({"ttg.memdesc_trans", "ttg.memdesc_reshape"})
# Ops through which a local_load's registers reach an MMA operand in the same block (scale chains: reshape / trans /
# layout conversion into a tensor-memory allocation).
_REG_FLOW = frozenset({"tt.reshape", "tt.trans", "ttg.convert_layout", "tt.bitcast", "ttng.tmem_alloc"})
_CLONE_OK = P._PURE | {"tt.get_program_id", "tt.get_num_programs", "arith.ceildivsi", "arith.floordivsi"}
_BITS = {"bf16": 16, "f16": 16, "f32": 32, "i8": 8, "i32": 32, "i64": 64, "f8E4M3FN": 8, "f8E5M2": 8, "i16": 16,
         "f64": 64, "i1": 8}


class Fail(Exception):
    pass


def _callee(line):
    m = P._FUNC_NAME.search(P._code(line))
    return m.group(1).strip('"') if m else ""


# Ops that synchronize (or may): an acquire / release, a barrier, an mbarrier, a fence, a TMA / tcgen05 op (mbarrier
# completion), griddepcontrol, any atomic (its ordering may be what makes a later read correct).
_SYNC_OP = re.compile(r"barrier|fence|atomic|^ttng\.|griddepcontrol|^nvvm\.|^llvm\.")

# Inline assembly, classified with the hoist pass's exact-template allowlist (``P._ALLOWLIST``) for the activation
# rules.  "quiet": orders no memory access of this CTA and waits for nothing (timer / SM-id reads, the PDL
# launch_dependents trigger, register-only conversions and warp reductions, the measurement stamp whose only effect is
# a plain weak store through its pointer operand - a WRITE, which R3 attributes); "spin": a readiness poll or the sync
# after it (kinds "poll" / "sync"), allowed in the warp-specialized region only as part of a recognized spin, which the
# activation producer clones; everything else - fences, CTA barriers, bulk-store drains, whole-spin templates, the PDL
# wait, the L2 prefetch, any allowlist entry not named here, and any assembly off the allowlist - synchronizes (or may).
_QUIET_TEMPLATES = frozenset({
    "griddepcontrol.launch_dependents; // dummy $0",
    "mov.u64 $0, %globaltimer;",
    "mov.u64 $0, %globaltimer; // $1",
    "mov.u32 $0, %smid;",
    "redux.sync.min.s32 $0, $1, 0xffffffff;",
    "{ .reg .u64 t; mov.u64 t, %globaltimer; st.global.u64 [$1], t; mov.u32 $0, 0; }",
    P._HELION_FP4X2_TO_F32,
    P._HELION_FP4X16_TO_F16,
    P._HELION_BF16X16_TO_F16,
    P._NVFP4_SCALE_GUARDED,
    P._NVFP4_SCALE,
    P._NVFP4_PACK_E2M1X2,
})
# Allowlist entries that synchronize (named so that a new allowlist entry must be classified: see the tests).
_SYNC_TEMPLATES = frozenset({
    "fence.acq_rel.gpu; // $0",
    P._READY_WAIT + "mov.u32 $0, 0; }",
    P._READY_WAIT + "fence.proxy.async.global; mov.u32 $0, 0; }",
    "bar.sync 0;",
    "griddepcontrol.wait; // dummy $0",
    "{ .reg .pred p; .reg .u32 t; .reg .u64 a; mov.u32 t, %tid.x; setp.lt.u32 p, t, {int}; mul.wide.u32 a, t, {int}; "
    "add.u64 a, a, $1; @p cp.async.bulk.prefetch.L2.global [a], {int}; mov.u32 $0, 0; }",
})


def _asm_class(line):
    """"quiet", "spin" or "sync" for an inline assembly op (see ``_QUIET_TEMPLATES``); "sync" off the allowlist."""
    entry = P._asm_entry(line)
    if entry is None:
        return "sync"
    if entry.kind in ("poll", "sync"):
        return "spin"
    return "quiet" if entry.template in _QUIET_TEMPLATES else "sync"


def _sync_functions(lines, funcs):
    """Names of the functions of the module that (transitively, through ``tt.call``) may synchronize: inline assembly
    that is not quiet (``_asm_class``, so any assembly off the allowlist), impure extern calls, atomics, volatile loads,
    barrier / mbarrier / fence / TMA / tcgen05 ops, or a call to a function the module does not define.  A readiness
    spin outlined into a noinline helper is one of these."""
    names = {f.name for f in funcs}
    direct, calls = set(), {}
    for f in funcs:
        cs = set()
        for line in lines[f.start + 1: f.end]:
            op = P._op(line)
            if op is None:
                continue
            if op == "tt.call":
                name = _callee(line)
                cs.add(name)
                if name not in names:
                    direct.add(f.name)
            elif op == P._ASM:
                if _asm_class(line) != "quiet":
                    direct.add(f.name)
            elif op == P._EXTERN:
                if not P._is_pure(line):
                    direct.add(f.name)
            elif op == "tt.load" and "isVolatile = true" in line:
                direct.add(f.name)
            elif _SYNC_OP.search(op):
                direct.add(f.name)
        calls[f.name] = cs
    sync, changed = set(direct), True
    while changed:
        changed = False
        for name, cs in calls.items():
            if name not in sync and cs & sync:
                sync.add(name)
                changed = True
    return sync


# Writes that stay in the generic proxy (the CTA-local ordering point after their root then orders them before the
# activation producer's cp.async reads); TMA / bulk stores are async-proxy writes: refused.
_GENERIC_WRITES = frozenset({"tt.store", "tt.atomic_rmw", "tt.atomic_cas"})
_ASYNC_WRITE = re.compile(r"descriptor_(store|reduce|scatter)|async_tma_|cp\.async\.bulk|cp\.reduce\.async\.bulk|"
                          r"\bst\.async|red\.async")


def _async_writing_functions(lines, funcs):
    """Functions that (transitively) may write other than through generic-proxy stores / atomics: an async-proxy store
    (TMA / bulk), inline assembly that is impure or may write (off the allowlist, or a stamp), an impure extern call,
    or a call to a function the module does not define."""
    names = {f.name for f in funcs}
    direct, calls = set(), {}
    for f in funcs:
        cs = set()
        for line in lines[f.start + 1: f.end]:
            op = P._op(line)
            if op == "tt.call":
                name = _callee(line)
                cs.add(name)
                if name not in names:
                    direct.add(f.name)
            elif op is not None and (_ASYNC_WRITE.search(op)
                                     or op == P._ASM and (not P._is_pure(line) or P._asm_may_write(line))
                                     or op == P._EXTERN and not P._is_pure(line)):
                direct.add(f.name)
        calls[f.name] = cs
    out, changed = set(direct), True
    while changed:
        changed = False
        for name, cs in calls.items():
            if name not in out and cs & out:
                out.add(name)
                changed = True
    return out


@dataclasses.dataclass(eq=False)
class Node:
    i: int
    op: str | None
    results: list
    uses: list
    regions: list = dataclasses.field(default_factory=list)
    seps: list = dataclasses.field(default_factory=list)   # line index closing each region
    end: int = -1
    parent: "Node | None" = None


def _parse_block(lines, a, b, parent):
    out, i = [], a
    while i < b:
        code = P._code(lines[i]).strip()
        if not code or code.startswith("^"):
            i += 1
            continue
        n = Node(i, P._op(lines[i]), P._results(lines[i]), P._uses(lines[i]), parent=parent)
        if code.endswith("{"):
            j = P._region_end(lines, i, b + 1)
            ind = P._indent(lines[i])
            seps = [k for k in range(i + 1, j) if P._indent(lines[k]) == ind and P._code(lines[k]).strip().startswith("}")]
            for s, e in zip([i] + seps, seps + [j]):
                n.regions.append(_parse_block(lines, s + 1, e, n))
            n.seps = seps + [j]
            n.end = j
            i = j + 1
        else:
            n.end = i
            i += 1
        out.append(n)
    return out


def _walk(nodes):
    for n in nodes:
        yield n
        for r in n.regions:
            yield from _walk(r)


def _ancestors(n):
    p = n.parent
    while p is not None:
        yield p
        p = p.parent


def _runs_ahead_into_later_tasks(lines, first, pre, spin_groups, w_spin_ops, roots, read_only):
    """Whether a weight producer that clones every spin in ``pre`` (the readiness spins before the region's first stage
    ``first``) still runs ahead of the consumer: some loop encloses ``first`` and every spin of ``pre`` (its later
    iterations are later tasks), and every spin the producer clones inside that loop polls only arguments no op of the
    kernel writes (``read_only``: the hoist pass's write analysis closed over the launch's storage classes), so the
    polled state was published before the launch or by another grid, never by this kernel's progress."""
    def polls_read_only(g):
        seeds = [x for x in _walk(g) if P._is_seed(lines[x.i])]
        polled = set()
        for x in seeds:
            for u in x.uses:
                u = u.split("#")[0]
                polled |= roots.get(u, frozenset({u}))
        return bool(seeds) and polled <= read_only

    for loop in _ancestors(first):
        if loop.op not in ("scf.for", "scf.while"):
            continue
        if not all(any(a is loop for a in _ancestors(g[0])) for g in pre):
            continue
        inside = [g for g in spin_groups if any(x.i in w_spin_ops for x in g)
                  and any(a is loop for a in _ancestors(g[0]))]
        return all(polls_read_only(g) for g in inside)
    return False


def _memdesc_shape(t):
    m = re.match(r"!ttg\.memdesc<([0-9x]+)x(\w+),\s*(.*)>$", t.strip())
    if not m:
        raise Fail(f"memdesc type {t}")
    return [int(d) for d in m.group(1).split("x")], m.group(2), m.group(3)


def _result_type(line):
    m = re.search(r"->\s*(!ttg\.memdesc<[^>]*>)", P._code(line))
    return m.group(1) if m else None


def _scalar_type(line):
    """Scalar result type of a single-result op (``tt.load`` of a scalar pointer: its pointee), else None."""
    if len(P._results(line)) != 1:
        return None
    code = P._code(line).strip()
    if P._op(line) == "tt.load":
        m = re.search(r":\s*!tt\.ptr<(i\d+|f\d+|bf16)>\s*$", code)
    else:
        m = re.search(r"(?:->|:)\s*(i\d+|f\d+|bf16)\s*$", code)
    return m.group(1) if m else None


def _ticket(lines, n, before, after, by_res, users, params):
    """Recognize a dynamic task loop  t = atomic_add(P, 1); while (t < ...) { ...; t = atomic_add(P, 1) }."""
    cond = n.regions[0][-1] if n.regions[0] else None
    yld = n.regions[1][-1] if n.regions[1] else None
    if cond is None or cond.op != "scf.condition" or yld is None or yld.op != "scf.yield":
        return None
    fwd = cond.uses[1:]
    if fwd != [a for a, _ in before] or len(after) != len(before):
        return None
    for k, (a, init) in enumerate(before):
        a0 = by_res.get(init)
        a1 = by_res.get(yld.uses[k]) if k < len(yld.uses) else None
        if a0 is None or a1 is None or a0.op != "tt.atomic_rmw" or a1.op != "tt.atomic_rmw":
            continue
        if a0.parent is not n.parent or a1.parent is not n or a0.uses[0] not in params:
            continue
        body = lambda x: P._code(lines[x.i]).split("=", 1)[1].strip()
        if body(a0) != body(a1) or "add," not in body(a0):
            continue
        if [u.i for u in users.get(a0.results[0], [])] != [n.i] or \
                [u.i for u in users.get(a1.results[0], [])] != [yld.i]:
            continue
        return dict(k=k, A0=a0, A1=a1, before=before, after=after, cond=cond, yld=yld, ptr=a0.uses[0])
    return None


_ARITH2 = {
    "addi": lambda a, b: a + b, "subi": lambda a, b: a - b, "muli": lambda a, b: a * b,
    "divsi": lambda a, b: int(a / b) if b else None, "remsi": lambda a, b: a - int(a / b) * b if b else None,
    "andi": lambda a, b: a & b, "ori": lambda a, b: a | b, "xori": lambda a, b: a ^ b,
    "maxsi": max, "minsi": min, "ceildivsi": lambda a, b: -(-a // b) if b else None,
    "floordivsi": lambda a, b: a // b if b else None,
    "divui": lambda a, b: a // b if b and a >= 0 else None, "remui": lambda a, b: a % b if b and a >= 0 else None,
}
_CMP = {"slt": lambda a, b: a < b, "sle": lambda a, b: a <= b, "sgt": lambda a, b: a > b, "sge": lambda a, b: a >= b,
        "eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "ult": lambda a, b: a < b, "ule": lambda a, b: a <= b,
        "ugt": lambda a, b: a > b, "uge": lambda a, b: a >= b}


def _scalar(lines, by_res, v, env, depth=0):
    """Value of scalar SSA ``v`` under ``env`` (loop IVs of expanded constant loops), or None if not constant."""
    v = v.split("#")[0]
    if v in env:
        return env[v]
    d = by_res.get(v)
    if d is None or depth > 64 or d.regions:
        return None
    code = P._code(lines[d.i]).split("=", 1)[-1].strip()
    m = re.match(r"arith\.constant (-?\d+|true|false) : (i\d+)\b", code)
    if m:
        return {"true": 1, "false": 0}.get(m.group(1), None) if m.group(1) in ("true", "false") else int(m.group(1))
    m = re.match(rf"arith\.(\w+) ({V}), ({V}) : i\d+$", code)
    if m and m.group(1) in _ARITH2:
        a, b = (_scalar(lines, by_res, x, env, depth + 1) for x in (m.group(2), m.group(3)))
        return None if a is None or b is None else _ARITH2[m.group(1)](a, b)
    m = re.match(rf"arith\.cmpi (\w+), ({V}), ({V}) : i\d+$", code)
    if m and m.group(1) in _CMP:
        a, b = (_scalar(lines, by_res, x, env, depth + 1) for x in (m.group(2), m.group(3)))
        return None if a is None or b is None else int(_CMP[m.group(1)](a, b))
    m = re.match(rf"arith\.select ({V}), ({V}), ({V}) : i\d+$", code)
    if m:
        c = _scalar(lines, by_res, m.group(1), env, depth + 1)
        return None if c is None else _scalar(lines, by_res, m.group(2) if c else m.group(3), env, depth + 1)
    m = re.match(rf"arith\.(?:extsi|extui|trunci|index_cast|index_castui) ({V}) : ", code)
    if m:
        return _scalar(lines, by_res, m.group(1), env, depth + 1)
    return None


_SEQ_LIMIT = 1 << 18


def _stage_words(lines, body, by_res, toks, tickets, bounds=None):
    """Stage sequence of one side of the ring protocol, as a word over the original loop / if nest.

    ``bounds`` maps a loop's line index to (lower, upper) bounds that replace its own: the word of a producer whose loop
    clone is re-bounded (rolled producers).

    ``toks`` maps the line index of a stage's counter anchor (the producer's last copy of the stage / the consumer's
    first slot view) to (ring label, scalar predicate or None).  Constant-bound loops are expanded (their IV is known,
    so the pipeliner's ``iv < X`` predicates evaluate); loops / ifs / ticket whiles with dynamic bounds or conditions
    stay symbolic and must match node for node (both partitions evaluate the same pure condition cone).  An
    unresolvable predicate stays in the word, so it can only match an identical predicated stage."""
    holders = set()
    by_i = {}
    for n in _walk(body):
        by_i[n.i] = n
    for i in toks:
        n = by_i[i]
        for a in _ancestors(n):
            holders.add(a.i)
    count = [0]

    def word(block, env):
        out = []
        for n in block:
            if n.i in toks:
                label, pred = toks[n.i]
                if pred is None:
                    out.append(("S", label))
                else:
                    v = _scalar(lines, by_res, pred, env)
                    if v is None:
                        out.append(("P", label, pred))
                    elif v:
                        out.append(("S", label))
                count[0] += 1
                if count[0] > _SEQ_LIMIT:
                    raise Fail("stage sequence too long to prove")
                continue
            if n.i not in holders:
                continue
            code = P._code(lines[n.i])
            if n.op == "scf.for":
                m = _FOR.match(code)
                lb, ub, st = (_scalar(lines, by_res, m.group(k), env) for k in ("lb", "ub", "st"))
                if bounds and n.i in bounds:
                    lb, ub = bounds[n.i]
                if None not in (lb, ub, st) and st > 0 and (ub - lb) // st <= 4096:
                    for x in range(lb, ub, st):
                        out.extend(word(n.regions[0], dict(env, **{m.group("iv"): x})))
                else:
                    out.append(("L", n.i, tuple(word(n.regions[0], env))))
            elif n.op == "scf.if":
                m = _IF.match(code)
                c = _scalar(lines, by_res, m.group("c"), env) if m else None
                els = n.regions[1] if len(n.regions) > 1 else []
                if c is None:
                    out.append(("I", n.i, tuple(word(n.regions[0], env)), tuple(word(els, env))))
                else:
                    out.extend(word(n.regions[0] if c else els, env))
            elif n.op == "scf.while" and n.i in tickets:
                out.append(("W", n.i, tuple(word(n.regions[1], env))))
            else:
                raise Fail(f"stage inside {n.op}")
        return out

    return word(body, {})


def _roll_window(lb, x, k, st):
    """Bounds of a rolled producer loop: K = num_stages - 1 peeled iterations earlier, up to the in-loop copy's predicate
    bound x (= ub - K * step), so the in-loop copy of tile iv + K * step issues tiles lb .. ub - step in order."""
    return lb - k * st, x


def _verified_rolls(roll_loops, producer_word, consumer_word, log=None):
    """The rolled loops whose rolled producer stage word equals the consumer's: each loop alone, then all of them
    together (none if the combination differs)."""
    ok = {}
    for i, v in roll_loops.items():
        if producer_word({i: v}) == consumer_word:
            ok[i] = v
        elif log is not None:
            log.append(f"loop at {i}: producer not rolled (rolled stage sequence differs)")
    if len(ok) > 1 and producer_word(ok) != consumer_word:
        if log is not None:
            log.append("rolled loops together change the stage sequence: none rolled")
        return {}
    return ok


def _rename(line, mapping):
    def sub(m):
        v = m.group(0)
        return mapping.get(v, v)
    code, sep, rest = line.partition(" loc(")
    return re.sub(r"%[\w$.-]+(?![\w$.-])(?!#)", sub, code) + sep + rest


def _first_difference(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k, x, y
    return min(len(a), len(b)), (a[len(b)] if len(a) > len(b) else None), (b[len(a)] if len(b) > len(a) else None)


def _enc_alignment(text, enc):
    """Byte alignment Triton's lowering assumes for a shared tile of encoding ``enc`` (alias or inline)."""
    m = re.match(r"\s*(#[\w.]+)", enc)
    if not m:
        return 1024
    name = m.group(1)
    if name.startswith("#ttg."):
        body = enc
    else:
        d = re.search(rf"^{re.escape(name)} = (.*)$", text, flags=re.M)
        if d is None:
            return 1024
        body = d.group(1)
    if "#ttg.swizzled_shared" in body:
        return 16
    return 1024   # nvmma_shared (TMA / tcgen05 / wgmma swizzle atoms) and anything unknown: the strictest


@dataclasses.dataclass(eq=False)
class Ring:
    alloc: Node
    name: str
    type: str
    depth: int
    tile: list
    elem: str
    enc: str
    copies: list = dataclasses.field(default_factory=list)
    views: list = dataclasses.field(default_factory=list)     # memdesc_index nodes on this ring
    deallocs: list = dataclasses.field(default_factory=list)

    @property
    def tile_bytes(self):
        return math.prod(self.tile) * _BITS[self.elem] // 8

    def slot_type(self):
        return f"!ttg.memdesc<{'x'.join(map(str, self.tile))}x{self.elem}, {self.enc}>"


def _param_writes(line, f, by_name, written_all, wild, f_classes):
    """Parameters (names) of ``f`` one of its ops may write, closed over ``f``'s storage classes: exactly the op's
    contribution to the hoist pass's ``_written_params`` (an unattributable target - a select, loop-carried or
    call-returned pointer, an integer address, assembly off the allowlist, an impure extern, a call outside the module
    or to a function that writes through an address it builds - writes every argument; the allowlisted stamp writes
    only its pointer operand's argument)."""
    if P._op(line) not in P._WRITES | {"tt.call", P._ASM, P._EXTERN}:
        return frozenset()
    h = P._op_write_hits(line, f, by_name, written_all, wild)
    idx = set(range(len(f.params))) if h is None else set(h)
    for members in f_classes:
        if idx & members:
            idx |= members
    return frozenset(f.params[k] for k in idx)


def _called(lines):
    """Names of the functions some ``tt.call`` of the module calls."""
    return {_callee(line) for line in lines if P._op(line) == "tt.call"}


def _storage_classes(lines, funcs, called, alias_classes):
    """Per function, the launch's storage alias classes as parameter-index sets, exactly as the hoist pass resolves
    them (``P.rewrite``: only for functions no op calls; None for a function whose classes name unknown parameters)."""
    classes = {}
    for g in funcs:
        if g.name in called:
            continue
        resolved = P._resolved_classes(g, alias_classes)
        classes[g.name] = resolved
    return classes


def _pdl_split(lines, funcs, f, body, called):
    """Line index after which the warp_specialize may start (``f.start``: anywhere).  The hoist pass's PDL rule: in a
    module that may wait on its PDL predecessor (Triton's ``gdc_wait``, assembly off the allowlist, an impure extern,
    a call outside the module, any other mention of ``griddepcontrol.wait``), both producers may read only after the
    predecessor finished, so the warp_specialize starts after the LAST top-level ``gdc_wait`` of the entry function (it
    then dominates both partitions); refused if there is none, if any assembly is off the allowlist (control may branch
    around the wait), or if the entry function has several blocks."""
    known = {g.name for g in funcs}
    if not (called - known or any(P._may_wait_for_predecessor(line) for line in lines)):
        return f.start
    if P._foreign_asm(lines):
        raise Fail("module may wait on its PDL predecessor: assembly off the allowlist")
    if any(P._op(line) in ("cf.br", "cf.cond_br", "cf.switch") for line in lines[f.start + 1: f.end]):
        raise Fail("module may wait on its PDL predecessor: entry function with several blocks")
    tops = [n for n in body if P._waits_for_predecessor(lines[n.i])]
    if not tops:
        raise Fail("PDL wait not at the entry function's top level")
    return tops[-1].i


def rewrite(text, shared=False, depth=None, producer_regs=None, maxnreg=None, log=None, pw=None, claim_early=False, only=None,
            group=1, late_ws=False, drop_waits=False, prefetch=False, neg=None, act=False, pwa=None,
            unroll=0, from_top=None, unroll_factor=0, alias_classes=(), roll=False):
    """``alias_classes``: kernel arguments (by name) that may share a storage at launch (the launch's facts, as for
    :func:`ttgir_weight_prefetch.rewrite`); an op writing one writes all.

    ``from_top`` k: only rings whose copies all lie in top-level ops k, k+1, ... of the entry function move (a
    suffix of the kernel's roots; implies ``late_ws``, so the earlier roots stay outside the warp_specialize).
    ``group`` G: one empty mbarrier (and one consumer release, i.e. one ``bar.sync``) per G consecutive stages of
    the global sequence (slots stay per stage, ready barriers stay per stage); ``late_ws``: the warp_specialize starts
    at the first top-level op that holds a ring stage (the ring-free prefix runs before the partitions exist);
    ``drop_waits``: delete consumer ``ttg.async_wait`` ops whose every user reads a moved slot; ``prefetch``: a
    ``ttg.local_load`` stage that is the only stage of a constant-bound loop is read one iteration ahead into a
    loop-carried register tile (wait for stage c+1 at the top of iteration c, load it and release stage c at the end),
    so the next slot's loads share a basic block with the current MMA chain.  ``neg="early"`` (TEST ONLY, a negative
    control that must produce wrong results): every stage is released right after its ready wait, before its slot is
    read (same number of releases).

    ``act``: rings whose source an earlier root of the kernel writes (activations exchanged through Helion's
    tile-dependency publication) move too, into a SECOND producer partition (``pwa`` warps, default = ``pw``) that
    clones every Helion readiness spin of the warp-specialized region at its original loop / if position, so it issues
    each copy only after performing every acquire the consumer had performed at that point; both producers wait on the
    same empty barrier and arrive on the same ready barrier (count = 32 x (pw + pwa)), so the consumer keeps no
    cp.async pipeline of its own (its now vestigial ``ttg.async_wait`` ops are dropped).  Implies ``late_ws``.
    Soundness of act (fail closed, checked on the original TTGIR): (R1) every recognized spin of the region preceding
    an activation copy is cloned; (R2) nothing else in the region synchronizes - no other impure inline asm, acquiring
    atomic or volatile load, and no ``tt.call`` to a function that (transitively) synchronizes (a spin outlined into a
    noinline helper would order the consumer's reads without being cloned); (R3) an activation is never written in the
    root that reads it, and if an EARLIER root of the region writes it (generic-proxy stores only), the consumer
    arrives on a CTA-local ordering mbarrier after that root and the activation producer waits on it before the reading
    root: the same-CTA program order the original read relied on is restored exactly (cross-CTA order stays with the
    cloned spins), with no reasoning about which counter or target covers which rows.
    ``neg="nowspin"`` (TEST ONLY): the weight producer omits the spins rule E cloned for its dependent loads (it reads
    the route table unordered and streams tiles of whatever it read).
    ``neg="nospin"`` / ``neg="noorder"`` (TEST ONLY): the activation producer omits the cloned spins / the ordering
    waits (must race).  ``unroll`` N: fully unroll every consumer loop
    with a ring stage, constant bounds, at most N iterations and no nested region (exact SSA substitution; ptxas does
    not unroll a loop whose body holds the mbarrier try-wait loop, while the canonical pipelined loop gets unrolled);
    ``unroll_factor`` F: such loops with more than ``unroll`` iterations (or every one, with ``unroll`` 0) are unrolled
    by F instead when F divides the trip count (the loop stays, F iterations per trip: bounded code growth).
    ``roll`` (ATT5): ROLLED PRODUCERS. Triton's pipeliner peels the first K = num_stages - 1 iterations' copies of a
    pipelined loop into K prologue copies (each a full tile copy: a 1-warp producer issues 64 cp.async per 32 KB tile),
    and copies iteration iv + K*step inside the loop under ``iv < ub - K*step``.  A producer partition does not need
    the peeling: it only issues the loop's tiles in order, waiting on empty slots.  With ``roll`` every such loop's
    producer clone starts K iterations early (lb - K*step) and runs to ub - K*step, so its in-loop copy issues the
    prologue's tiles too and the K prologue copies are not emitted: the same tiles, in the same global stage order,
    into the same slots, with ~1/(K+1) of the producer's copy code.  Applied only when it is provably the peeling: the
    loop's producer stages are exactly K unpredicated prologue stages + one in-loop stage predicated by
    ``iv < ub - K*step`` (constant bounds), the producer cone uses the induction variable only through that predicate
    and ``iv + K*step``, and no other stage is issued between the first prologue copy and the loop.  Motivation:
    instruction-cache retention across launches (att5 NOTES E1c): a megakernel whose code outgrows what the GPU's
    instruction caches keep between launches runs its roots with cold code every launch."""
    if neg not in (None, "early", "nospin", "noorder", "nowspin"):
        raise Fail(f"unknown negative control {neg}")
    if neg in ("nospin", "noorder") and not act:
        raise Fail(f"negative control {neg} needs act")
    if act or from_top is not None:
        late_ws = True
    if neg == "early" and group != 1:
        raise Fail("negative control needs group 1")
    text = P._uniquify(text)
    lines = text.splitlines()
    funcs = P._functions(lines)
    entry = [f for f in funcs if re.search(r"tt\.func\s+public", lines[f.start])]
    if len(entry) != 1:
        raise Fail("expects one public function")
    f = entry[0]
    if "ttg.warp_specialize" in text:
        raise Fail("already warp specialized")
    if re.search(r'\bttg\.maxnreg"?\s*=', text.split("tt.func", 1)[0]):
        # AllocateWarpGroups splits a module register cap between the partitions on its own (the producer gets 24
        # registers and spills); its setmaxnreg sequence is not ours to prove, so refuse.
        raise Fail("module has ttg.maxnreg")
    for g in funcs:   # every function of the module, not only the entry (callees run in the default partition)
        reason = P._unsupported(lines, g)
        if reason:
            raise Fail(reason)
    called = _called(lines)
    if f.name in called:
        raise Fail("called entry function (its parameters may alias)")
    classes = _storage_classes(lines, funcs, called, alias_classes)
    if classes.get(f.name) is None:
        raise Fail("argument storage classes name unknown parameters")
    classes = {k: v for k, v in classes.items() if v is not None}
    nw = int(re.search(r'"ttg.num-warps" = (\d+)', text).group(1))
    body = _parse_block(lines, f.start + 1, f.end - 1, None)
    # PDL: the producers may read only after the predecessor finished (line index after which the warp_specialize
    # starts; the prefix stays outside)
    split_at = _pdl_split(lines, funcs, f, body, called)
    nodes = list(_walk(body))
    by_res = {r: n for n in nodes for r in n.results}
    region_vals = {}
    for n in nodes:
        m = _FOR.match(P._code(lines[n.i]))
        if m:
            region_vals[m.group("iv")] = ("iv", n)
            for a in re.findall(rf"({V})\s*=", m.group("ia") or ""):
                region_vals[a] = ("iter", n)
    users = {}
    for n in nodes:
        for u in n.uses:
            users.setdefault(u.split("#")[0], []).append(n)
    tickets = {}
    for n in nodes:
        if n.op != "scf.while":
            continue
        m = _WHILE.match(P._code(lines[n.i]))
        if m is None or len(n.regions) != 2:
            raise Fail("unrecognized scf.while")
        before = re.findall(rf"({V})\s*=\s*({V})", m.group("args"))
        for a, _ in before:
            region_vals[a] = ("wbefore", n)
        bb = lines[n.seps[0] + 1]
        after = re.findall(r"(%[\w$.-]+)\s*:", P._code(bb)) if bb.strip().startswith("^bb") else []
        for a in after:
            region_vals[a] = ("wafter", n)
        tk = _ticket(lines, n, before, after, by_res, users, f.params)
        if tk is not None:
            tickets[n.i] = tk
    # which arguments are written: the hoist pass's analysis, closed over the storage classes exactly as it is there
    written_all, wild = P._write_analysis(lines, funcs, classes)
    written = written_all[f.name]
    read_only = {p for k, p in enumerate(f.params) if k not in written}
    by_name = {g.name: g for g in funcs}
    f_classes = classes.get(f.name, [])

    def write_hits(n):
        return _param_writes(lines[n.i], f, by_name, written_all, wild, f_classes)
    # ---------------- rings
    rings = {}
    for n in nodes:
        if n.op == "ttg.local_alloc" and not n.uses:
            t = _result_type(lines[n.i])
            if t and "mutable" in t:
                shape, elem, enc = _memdesc_shape(t)
                if len(shape) >= 2 and elem in _BITS:
                    rings[n.results[0]] = Ring(n, n.results[0], t, shape[0], shape[1:], elem, enc.replace(", mutable", "").strip() + ", mutable")
    view_ring = {}
    for n in nodes:
        if n.op == "ttg.memdesc_index" and n.uses and n.uses[0] in rings:
            rings[n.uses[0]].views.append(n)
            view_ring[n.results[0]] = rings[n.uses[0]]
        if n.op == "ttg.local_dealloc" and n.uses and n.uses[0] in rings:
            rings[n.uses[0]].deallocs.append(n)
    copy_of = {}
    for n in nodes:
        if n.op == P._COPY:
            m = _COPY_RE.match(P._code(lines[n.i]))
            if not m:
                raise Fail("unrecognized copy")
            if m.group("dst") in view_ring:
                view_ring[m.group("dst")].copies.append(n)
                copy_of[n.i] = m
    # TMA copies (Triton's pipeliner: barrier_expect bar[s], bytes; TMA desc[coords] -> slot, bar[s]; consumer
    # wait_barrier bar[s]) into a multibuffer view: ring stages too, issued by the weight producer with the ring's ready
    # barrier as their completion barrier.
    tma_of = {}
    for n in nodes:
        if n.op == P._TMA_COPY:
            m = _TMA_RE.match(P._code(lines[n.i]))
            if not m:
                raise Fail("unrecognized TMA copy")
            if m.group("dst") in view_ring:
                view_ring[m.group("dst")].copies.append(n)
                tma_of[n.i] = m

    def dst_of(c):
        return (tma_of[c.i] if c.i in tma_of else copy_of[c.i]).group("dst")

    sel = []
    act_names, act_roots = set(), {}
    top_index = {x.i: k for k, n in enumerate(body) for x in _walk([n])}
    for r in rings.values():
        if not r.copies:
            continue
        if from_top is not None and any(top_index[c.i] < from_top for c in r.copies):
            continue
        if len({c.i in tma_of for c in r.copies}) != 1:
            raise Fail(f"ring {r.name} filled by both TMA and cp.async copies")
        roots = set()
        for c in r.copies:
            src = tma_of[c.i].group("desc") if c.i in tma_of else copy_of[c.i].group("src")
            roots |= f.roots.get(src, frozenset({src}))
        if not roots <= read_only:
            if r.copies[0].i in tma_of:
                if log is not None:
                    log.append(f"ring {r.name}: TMA source written in-kernel, stays")
                continue
            if not act:
                if log is not None:
                    log.append(f"ring {r.name}: source written in-kernel, stays")
                continue
            if not roots <= set(f.params):
                raise Fail(f"activation ring {r.name} reads a non-argument pointer")
            act_names.add(r.name)
            act_roots[r.name] = frozenset(roots)
        for u in users.get(r.name, []):
            if u.op not in ("ttg.memdesc_index", "ttg.local_dealloc"):
                raise Fail(f"ring {r.name} used by {u.op}")
        sel.append(r)
    if only is not None:
        sel = [r for k, r in enumerate(sel) if k in only]
        act_names &= {r.name for r in sel}
    if not sel:
        raise Fail("no movable ring")
    if act_names and tickets:
        raise Fail("activation rings with dynamic ticket loops")
    if pw == "auto":
        pw = 1 if act_names else None
    if pwa == "auto":
        pwa = 1
    D = depth or max(r.depth for r in sel)
    sel_names = {r.name for r in sel}
    copies = sorted((c for r in sel for c in r.copies), key=lambda c: c.i)
    copy_set = {c.i for c in copies}

    # ---------------- producer stages + cones
    keep = set()
    captures = []
    pcaps = []   # act mode: non-clonable scalars of the function's top level (must lie in the ring-free prefix)
    dep_loads = {}   # rule E: line of a load the producers clone -> the kernel arguments it reads

    def need(v):
        v = v.split("#")[0]
        if v in f.params:
            if v not in captures:
                captures.append(v)
            return
        if v in region_vals:
            kind, owner = region_vals[v]
            if kind in ("wbefore", "wafter"):
                tk = tickets.get(owner.i)
                k = tk and tk["k"]
                if tk is None or v not in (tk["before"][k][0], tk["after"][k]):
                    raise Fail(f"producer cone reads while-carried {v}")
            elif kind != "iv":
                raise Fail(f"producer cone reads loop-carried {v}")
            mark(owner)
            return
        d = by_res.get(v)
        if d is None:
            raise Fail(f"unknown value {v}")
        if d.i in keep:
            return
        if d.op not in _CLONE_OK:
            if (act_names or tma_of) and d.parent is None and _scalar_type(lines[d.i]):
                # e.g. the per-launch epoch Helion loads at entry: captured by the warp_specialize (checked below to
                # be defined before it), so every partition sees the very same SSA value
                if v not in pcaps:
                    pcaps.append(v)
                return
            if d.op == "tt.load" and tma_of:
                # RULE E (dependent addresses, e.g. the expert of a routed weight tile): a load the producer clones.
                # Sound only if the producer reads the same bytes as the consumer: checked after the cone is known
                # (no writer of the loaded arguments inside the warp-specialized region; the producers clone every
                # readiness spin that precedes the load, so they perform the consumer's acquires first).
                if "isVolatile = true" in lines[d.i] or not d.uses:
                    raise Fail("producer cone has a volatile load")
                roots_ = f.roots.get(d.uses[0], frozenset({d.uses[0]}))
                if not roots_ <= set(f.params):
                    raise Fail("producer cone loads through a non-argument pointer")
                dep_loads[d.i] = frozenset(roots_)
            elif _generic_op(lines[d.i]) in ("tt.reduce", "tt.scan") and tma_of:
                # integer reductions / scans with a pure combiner: cloned verbatim (warp-count independent)
                tys = P._code(lines[d.end]).rsplit(":", 1)[-1]
                if re.search(r"\b(f16|bf16|f32|f64|f8E\w+)\b", tys):
                    raise Fail("producer cone has a floating-point reduction")
                if not d.regions:
                    raise Fail("producer cone reduction without a combiner region")
                for x in _walk(d.regions[0]):
                    if x.op not in P._PURE | {"tt.reduce.return", "tt.scan.return"}:
                        raise Fail(f"producer cone reduction combiner has {x.op}")
            else:
                raise Fail(f"producer cone has {d.op or P._code(lines[d.i]).strip()[:80]}")
        keep.add(d.i)
        for u in d.uses:
            need(u)
        for a in _ancestors(d):
            mark(a)

    def mark(owner):
        if owner.i in keep:
            return
        keep.add(owner.i)
        code = P._code(lines[owner.i])
        if owner.op == "scf.for":
            m = _FOR.match(code)
            for k in ("lb", "ub", "st"):
                need(m.group(k))
        elif owner.op == "scf.if":
            m = _IF.match(code)
            if m is None:
                raise Fail("unrecognized scf.if")
            need(m.group("c"))
        elif owner.op == "scf.while" and owner.i in tickets:
            tk = tickets[owner.i]
            # the claim atomics move to the producer; the consumer receives tickets through the mailbox
            for a in (tk["A0"], tk["A1"]):
                keep.add(a.i)
                for u in a.uses:
                    need(u)
            need(tk["cond"].uses[0])
        else:
            raise Fail(f"copy nested in {owner.op}")
        for a in _ancestors(owner):
            mark(a)

    def scalar_pred(mask):
        if mask is None:
            return None
        d = by_res.get(mask)
        if d is None:
            raise Fail("mask is a region value")
        code = P._code(lines[d.i])
        if d.op == "arith.constant" and "dense<true>" in code:
            return None
        if d.op == "tt.splat" and re.search(r":\s*i1\s*->", code):
            return d.uses[0]
        raise Fail("copy mask is not a splat predicate")

    def split_mask(mask):
        """(scalar pipeliner predicate or None, residual user mask or None)."""
        if mask is None:
            return None, None
        d = by_res.get(mask)
        if d is not None and d.op == "arith.constant" and "dense<false>" in P._code(lines[d.i]):
            return DEAD, None   # a prologue copy of an iteration that does not exist: no stage
        if d is not None and d.op == "arith.andi":
            for a_, b_ in ((0, 1), (1, 0)):
                e = by_res.get(d.uses[a_])
                if e is not None and e.op == "tt.splat" and re.search(r":\s*i1\s*->", P._code(lines[e.i])):
                    return e.uses[0], d.uses[b_]
        try:
            return scalar_pred(mask), None
        except Fail:
            return None, mask

    preds, resid_of = {}, {}
    DEAD = "<dead>"

    def _in_pipelined_loop(c):
        return "tt.num_stages" in lines[c.parent.end] or "loop.stage" in lines[c.i]

    def _pipelined_by_triton(c):
        """The copy's loop is one Triton software-pipelined (more than one stage, or the copy carries a stage)."""
        mm = re.search(r"tt\.num_stages = (\d+)", lines[c.parent.end])
        return (mm is not None and int(mm.group(1)) > 1) or "loop.stage" in lines[c.i]

    def const_i1(v):
        d = by_res.get(v)
        if d is None or d.op != "arith.constant":
            return None
        mm = re.search(r"arith\.constant (true|false)\b", P._code(lines[d.i]))
        return None if mm is None else mm.group(1) == "true"

    for c in copies:
        if c.i in tma_of:
            m = tma_of[c.i]
            need(m.group("desc"))
            for x in re.findall(V, m.group("coords")):
                need(x)
            k1 = const_i1(m.group("pred"))
            p = None if k1 is True else DEAD if k1 is False else m.group("pred")
            if p is None and c.parent is not None and c.parent.op == "scf.for" and _pipelined_by_triton(c):
                raise Fail("in-loop TMA copy without a pipeliner predicate")
            preds[c.i] = p
            resid_of[c.i] = None
            if p is not None and p is not DEAD:
                need(p)
            for a in _ancestors(c):
                mark(a)
            continue
        m = copy_of[c.i]
        if m.group("other"):
            need(m.group("other"))
        need(m.group("src"))
        p, resid = split_mask(m.group("mask"))
        if p is None and resid is not None and c.parent is not None and c.parent.op == "scf.for" \
                and any(x is c.parent for x in [c.parent]) and _in_pipelined_loop(c):
            raise Fail("in-loop copy without a pipeliner predicate")
        preds[c.i] = p
        resid_of[c.i] = resid
        if p is not None and p is not DEAD:
            need(p)
        if resid is not None:
            need(resid)
        for a in _ancestors(c):
            mark(a)

    def slot_of_copy(c):
        return by_res[dst_of(c)].uses[1]

    pstages = {}
    for c in copies:
        if preds[c.i] is DEAD:
            continue   # copy of an iteration that does not exist (constant-false mask): not a stage
        pstages.setdefault((id(c.parent), slot_of_copy(c)), []).append(c)
    for cs in pstages.values():
        if len({preds[c.i] for c in cs}) != 1:
            raise Fail("copies of one stage with different predicates")
        if len({view_ring[dst_of(c)].name for c in cs}) != len(cs):
            raise Fail("two copies of one stage into one ring")

    # ---------------- TMA stages: Triton's per-stage completion barriers (one barrier_expect + the stage's TMA copies
    # + the consumer's wait_barrier per slot) are replaced by the ring's ready barrier; every use of such a barrier
    # array must be one of those (else the barrier orders something else: refuse).
    tma_bars = {}      # barrier array name -> Ring record of the barrier allocation
    tma_bytes = {}     # stage key -> bytes of its TMA members (Triton's barrier_expect count, verified)
    ring_has_tma = any(c.i in tma_of for c in copies)
    for c in copies:
        if c.i not in tma_of:
            continue
        bv = tma_of[c.i].group("bar")
        br = view_ring.get(bv)
        if br is None or br.tile != [1] or br.elem != "i64":
            raise Fail("TMA copy completion barrier is not a view of a barrier array")
        tma_bars[br.name] = br
    bar_drop = set()   # lines dropped from the consumer: the arrays, their views, init / inval / expect / waits
    for bname, br in tma_bars.items():
        bar_drop.add(br.alloc.i)
        bar_drop.update(d.i for d in br.deallocs)
        for u in users.get(bname, []):
            if u.op not in ("ttg.memdesc_index", "ttg.local_dealloc"):
                raise Fail(f"TMA barrier array used by {u.op}")
        for ix in br.views:
            bar_drop.add(ix.i)
            for u in users.get(ix.results[0], []):
                if u.op in ("ttng.init_barrier", "ttng.inval_barrier", "ttng.barrier_expect", "ttng.wait_barrier"):
                    if u.op == "ttng.wait_barrier" and " deps " in P._code(lines[u.i]):
                        raise Fail("TMA barrier wait with dependencies")
                    bar_drop.add(u.i)
                elif u.op == P._TMA_COPY:
                    if u.i not in copy_set:
                        raise Fail("TMA barrier shared with a copy that stays")
                else:
                    raise Fail(f"TMA barrier view used by {u.op}")
    for key, cs in pstages.items():
        tm = [c for c in cs if c.i in tma_of]
        if not tm:
            continue
        tb = sum(view_ring[dst_of(c)].tile_bytes for c in tm)
        bars = {tma_of[c.i].group("bar") for c in tm}
        if len(bars) != 1:
            raise Fail("TMA copies of one stage complete on different barriers")
        bv = bars.pop()
        exps = [u for u in users.get(bv, []) if u.op == "ttng.barrier_expect"]
        if len(exps) != 1:
            raise Fail("TMA stage without exactly one barrier_expect")
        em = _EXPECT_RE.match(P._code(lines[exps[0].i]))
        if em is None or int(em.group("bytes")) != tb or em.group("pred") != tma_of[tm[0].i].group("pred"):
            raise Fail("barrier_expect bytes / predicate differ from the stage's TMA copies")
        tma_bytes[key] = tb

    # ---------------- consumer stages
    PASSIVE = ("scf.for", "scf.yield", "ttng.wait_barrier", "scf.if")

    def carried_to_k(y, k):
        """Values the k-th operand of scf.yield ``y`` becomes: the region op's k-th result and, for a loop, the k-th
        iter_arg of the next iteration."""
        owner = y.parent
        if owner is None:
            raise Fail("slot view yielded at function level")
        code = P._code(lines[owner.i])
        res = _FOR.match(code) if owner.op == "scf.for" else _IF.match(code) if owner.op == "scf.if" else None
        if res is None:
            raise Fail(f"slot view yielded by {owner.op}")
        out = []
        if res.group("res"):
            out.append(f"{res.group('res')}#{k}" if res.group("n") else res.group("res"))
        if owner.op == "scf.for":
            ia = re.findall(rf"({V})\s*=", res.group("ia") or "")
            if k >= len(ia):
                raise Fail("yield / iter_args mismatch")
            out.append(ia[k])
        return out

    def check_passive(v, carried=False, seen=None):
        """A slot view may flow through scf.yield only into ttng.wait_barrier dependency operands (Triton 3.8's MMAv5
        pipelining carries MMA operand views to the completion wait); any other reader of a carried view would read
        the slot after its release."""
        seen = seen if seen is not None else set()
        if v in seen:
            return
        seen.add(v)
        pat = re.compile(re.escape(v) + (r"(?![\w$.#-])" if "#" in v else r"(?![\w$.-])(?!#)"))
        cands = users.get(v.split("#")[0], [])
        for u in cands:
            code = P._code(lines[u.i])
            rhs = code.split("=", 1)[1] if P._RESULTS.match(code) else code
            if not pat.search(rhs):
                continue   # a use of another result of the same multi-result op
            if u.op == "scf.yield":
                yv = [w for w in re.findall(rf"{V}", rhs.split(":")[0])]
                for k, w in enumerate(yv):
                    if w == v:
                        for c in carried_to_k(u, k):
                            check_passive(c, True, seen)
            elif u.op == "ttng.wait_barrier":
                head, _, deps = rhs.partition(" deps ")
                if pat.search(head.split(":")[0]):
                    raise Fail("slot view used as a barrier operand")
            elif u.op in ("scf.for", "scf.if") or u.i in copy_set:
                continue
            elif carried:
                raise Fail(f"carried slot view read by {u.op} after its yield")

    cstages = {}
    for r in sel:
        for ix in r.views:
            us = [u for u in users.get(ix.results[0], []) if u.i not in copy_set]
            if us:
                cstages.setdefault((id(ix.parent), ix.uses[1]), []).append(ix)
    cinfo = {}
    for key, ixs in cstages.items():
        work, views, readers = [ix.results[0] for ix in ixs], set(), []
        while work:
            v = work.pop()
            views.add(v)
            check_passive(v)
            for u in users.get(v, []):
                if u.i in copy_set or u.op in PASSIVE:
                    continue
                if u.op == "ttg.memdesc_trans" or (u.op in _SLOT_VIEWS and ring_has_tma):
                    work.append(u.results[0])
                elif u.op in ("ttg.local_load", "ttng.tc_gen5_mma") or (u.op in _MMA_OPS and ring_has_tma):
                    if u.op in _MMA_OPS and v not in u.uses[:2]:
                        raise Fail("moved slot as MMA accumulator or scale")
                    readers.append(u)
                else:
                    raise Fail(f"moved slot read by {u.op}")
        if not readers:
            raise Fail("stage without readers")
        if any(rd.parent is not ixs[0].parent for rd in readers):
            raise Fail("stage readers in another block")
        readers = sorted(readers, key=lambda x: x.i)
        kinds = {"ttng.tc_gen5_mma" if rd.op in _MMA_OPS else rd.op for rd in readers}
        if len(kinds) != 1:
            # Mixed stage (e.g. an NVFP4 weight tile read by a scaled MMA and its scale tile by a local_load that
            # feeds the MMA's tensor-memory scale operand): released by tcgen05.commit after its last MMA iff no
            # generic read of the stage is outstanding once that MMA is issued, i.e. the last reader is an MMA and
            # every local_load's registers reach the operands of an MMA reader issued no later than it (the load has
            # then completed before that MMA, hence before the commit).
            mmas = [rd for rd in readers if rd.op in _MMA_OPS]
            if not ring_has_tma or readers[-1].op not in _MMA_OPS:
                raise Fail("stage read by local_load and MMA")
            p0 = readers[0].parent
            blk = p0.regions[[k for k, r in enumerate(p0.regions) if readers[0] in r][0]] if p0 else body
            pos = {x.i: k for k, x in enumerate(blk)}
            for ld in (rd for rd in readers if rd.op == "ttg.local_load"):
                vals, ok_ = set(ld.results), False
                for x in blk[pos[ld.i] + 1: pos[readers[-1].i] + 1]:
                    if x.op in _MMA_OPS and x in mmas and set(x.uses) & vals:
                        ok_ = True
                        break
                    if x.op in _REG_FLOW and set(x.uses) & vals:
                        vals |= set(x.results)
                if not ok_:
                    raise Fail("stage's local_load does not feed an MMA reader before its release")
        cinfo[key] = (sorted(ixs, key=lambda x: x.i), readers, kinds.pop() if len(kinds) == 1 else "ttng.tc_gen5_mma")

    # ---------------- stage-count invariant: per pipelined loop instance the producer issues exactly as many stages
    # as the consumer reads (prologue + live in-loop copies == peeled reads + one read per iteration)
    def const_of(v):
        d = by_res.get(v)
        if d is None or d.op != "arith.constant":
            return None
        mm = re.search(r"arith\.constant (-?\d+) : i(?:32|64)\b", P._code(lines[d.i]))
        return int(mm.group(1)) if mm else None

    def siblings_of(n):
        return n.parent.regions[[k for k, r in enumerate(n.parent.regions) if n in r][0]] if n.parent else body

    p_anchor = {key: sorted(cs, key=lambda c: c.i)[-1] for key, cs in pstages.items()}
    c_anchor = {key: v[0][0] for key, v in cinfo.items()}
    ring_of_p = {key: {view_ring[dst_of(c)].name for c in cs} for key, cs in pstages.items()}
    ring_of_c = {key: {ix.uses[0] for ix in v[0]} for key, v in cinfo.items()}

    def contains(n, x):
        while x is not None:
            if x is n:
                return True
            x = x.parent
        return False
    loops_checked = set()
    roll_loops = {}   # roll: pipelined loop line -> (producer lb, producer ub, prologue copy lines not emitted)
    for key, a in p_anchor.items():
        L = a.parent
        if L is None or L.op != "scf.for" or L.i in loops_checked:
            continue
        if const_of(key[1]) is not None:
            continue  # constant slot: a prologue stage (counted with its pipelined loop)
        loops_checked.add(L.i)
        rs = ring_of_p[key]
        sib = siblings_of(L)
        idx = sib.index(L)
        def touches(n):
            return any(contains(n, x) for k2, x in p_anchor.items() if ring_of_p[k2] & rs) or \
                any(contains(n, x) for k2, x in c_anchor.items() if ring_of_c[k2] & rs)
        lo = idx
        while lo > 0 and not (sib[lo - 1].regions and touches(sib[lo - 1])):
            lo -= 1
        hi = idx
        while hi + 1 < len(sib) and not (sib[hi + 1].regions and touches(sib[hi + 1])):
            hi += 1
        region = set(id(x) for x in sib[lo:hi + 1] if x is not L)
        pp = [k2 for k2, x in p_anchor.items() if ring_of_p[k2] & rs and id(x) in region]
        cp = [k2 for k2, x in c_anchor.items() if ring_of_c[k2] & rs and id(x) in region]
        pin = [k2 for k2, x in p_anchor.items() if ring_of_p[k2] & rs and x.parent is L]
        cin = [k2 for k2, x in c_anchor.items() if ring_of_c[k2] & rs and x.parent is L]
        if any(const_of(k2[1]) is not None for k2 in pin):
            raise Fail("constant-slot stage inside a pipelined loop")
        if len(pin) != 1 or len(cin) != 1:
            raise Fail("pipelined loop with other than one producer and one consumer stage per iteration")
        mf = _FOR.match(P._code(lines[L.i]))
        lb, ub, st = (const_of(mf.group(g)) for g in ("lb", "ub", "st"))
        if None in (lb, ub, st) or st <= 0:
            raise Fail("pipelined loop with dynamic bounds (stage count unprovable)")
        n = max(0, -(-(ub - lb) // st))
        p = preds[p_anchor[pin[0]].i]
        if any(preds[p_anchor[k2].i] is not None for k2 in pp):
            raise Fail("predicated prologue stage")
        if p is None:
            live = n
        else:
            d = by_res.get(p)
            mm = re.match(rf"^\s*{V}\s*=\s*arith\.cmpi slt, ({V}), ({V})", P._code(lines[d.i])) if d else None
            if mm is None or mm.group(1) != mf.group("iv") or const_of(mm.group(2)) is None:
                raise Fail("in-loop stage predicate is not iv < constant")
            x = const_of(mm.group(2))
            live = sum(1 for k2 in range(n) if lb + k2 * st < x)
        if len(pp) + live != len(cp) + n:
            raise Fail(f"stage counts differ (producer {len(pp)}+{live}, consumer {len(cp)}+{n})")
        if roll and pp and p is not None:
            K = len(pp)
            iv = mf.group("iv")
            pro = {c.i for k2 in pp for c in pstages[k2]}
            first = min(pro)
            ok = x == ub - K * st and max(pro) < L.i
            # the producer cone reads iv only through the predicate and iv + K*step (the peeled tile index)
            for nd in _walk(L.regions[0]):
                if nd.i not in keep or iv not in [u.split("#")[0] for u in nd.uses]:
                    continue
                if nd is d:
                    continue
                if not (nd.op == "arith.addi" and len(nd.uses) == 2 and nd.uses[0] == iv
                        and const_of(nd.uses[1]) == K * st):
                    ok = False
            # no other stage between the first prologue copy and the loop (the global order must not change)
            for k2, xa in p_anchor.items():
                if k2 not in pp and k2 not in pin and first < xa.i < L.i:
                    ok = False
            if ok:
                roll_loops[L.i] = (*_roll_window(lb, x, K, st), pro)
            elif log is not None:
                log.append(f"loop at {L.i}: producer not rolled")

    # ---------------- global stage-sequence guard: both partitions advance ONE counter over all moved rings, so for
    # every count the producer must fill exactly the set of rings the consumer reads (the per-loop counts above
    # cannot see how stages of different rings interleave: mixed pipeline distances diverge at count 2 and hang).
    p_toks = {p_anchor[k].i: (frozenset(ring_of_p[k]), preds[p_anchor[k].i]) for k in pstages}
    c_toks = {c_anchor[k].i: (frozenset(ring_of_c[k]), None) for k in cinfo}
    pw_ = _stage_words(lines, body, by_res, p_toks, tickets)
    cw_ = _stage_words(lines, body, by_res, c_toks, tickets)
    if pw_ != cw_:
        k, a, b = _first_difference(pw_, cw_)
        raise Fail(f"stage sequences differ at position {k} (producer {a}, consumer {b})")
    # rolled producers (ATT6): the guard again on the ROLLED producer (its loop clone re-bounded, the peeled prologue
    # copies not emitted), per loop and then for all rolled loops together. A roll whose producer word is not the
    # consumer's (a nested ring-bearing loop, a second ring's stages in the loop, ...) is refused, not the ring.
    if roll_loops:
        roll_loops = _verified_rolls(roll_loops, lambda rl: _stage_words(
            lines, body, by_res, {i: v for i, v in p_toks.items() if not any(i in pro for _lb, _ub, pro in rl.values())},
            tickets, bounds={i: (lb_, ub_) for i, (lb_, ub_, _pro) in rl.items()}), cw_, log)

    # nodes (loops / ifs) whose region contains a stage
    def counter_nodes(stage_nodes):
        out = set()
        for n in stage_nodes:
            for a in _ancestors(n):
                if a.op not in ("scf.for", "scf.if") and not (a.op == "scf.while" and a.i in tickets):
                    raise Fail(f"stage inside {a.op}")
                out.add(a.i)
        return out
    c_ctr = counter_nodes([ixs[0] for ixs, _, _ in cinfo.values()])

    # ---------------- activation rings (act) and dependent producer cones (rule E): spins to clone into the producers
    # + soundness rules
    spin_ops, spin_groups, split_est = set(), [], None
    w_spin_ops = set()   # rule E: spins the weight producer clones (the activation producer clones them too)
    order_after = {}   # act: reading root (top-level line) -> the last earlier root of the region writing its activation

    def top_of(n):
        while n.parent is not None:
            n = n.parent
        return n

    def clone_spins(anchor_lines, into):
        """Clone every recognized spin of the region that precedes one of ``anchor_lines`` in program order or shares
        a loop with one: the cloning producer then has performed every acquire the consumer had at each anchor."""
        last = max(anchor_lines)
        loops = {a.i for i in anchor_lines for a in _ancestors(by_line[i]) if a.op == "scf.for"}
        for grp in spin_groups:
            if grp[0].i <= split_est:
                continue
            if not (grp[0].i < last or any(a.op == "scf.for" and a.i in loops for a in _ancestors(grp[0]))):
                continue
            defs, toks = set(), set()
            for x in grp:
                if x.op in (P._ASM, "scf.while"):
                    into.add(x.i)
                keep.add(x.i)
                for k in range(x.i, x.end + 1):
                    code = P._code(lines[k])
                    defs |= set(re.findall(rf"({V})(?::\d+)?\s*=(?!=)", code))
                    if code.strip().startswith("^"):
                        defs |= set(re.findall(rf"({V})\s*:(?!\d)", code))
                    toks |= {t.split("#")[0] for t in re.findall(V, code)}
            for t in sorted(toks - {d.split("#")[0] for d in defs}):
                need(t)
            for a in _ancestors(grp[0]):
                mark(a)
    by_line = {n.i: n for n in nodes}
    if act_names or dep_loads:
        marks = set(copy_set) | {ix.i for r in sel for ix in r.views} | set(c_ctr) | \
            {rd.i for _, rds, _ in cinfo.values() for rd in rds}
        for n in body:
            if n.i > split_at and any(n.i <= k <= n.end for k in marks):
                split_est = n.i - 1
                break
        if split_est is None:
            raise Fail("no warp-specialized region")
        # recognized Helion readiness spins: ld.acquire.gpu seed(s) [+ pure], scf.while polling them, bar.warp.sync
        spin_lines = set()
        for n in nodes:
            if n.i in spin_lines or not P._is_seed(lines[n.i]):
                continue
            k = P._wait_at(lines, n.i, f.end)
            if k is None:
                continue
            sib = siblings_of(n)
            grp = [x for x in sib[sib.index(n):] if x.i < k]
            if not grp or grp[-1].end != k - 1:
                raise Fail("unrecognized spin layout")
            spin_groups.append(grp)
            for x in grp:
                spin_lines.update(range(x.i, x.end + 1))
        # (R2) inside the region nothing else may synchronize: the consumer's reads were ordered only by those spins
        sync_funcs = _sync_functions(lines, funcs)
        for n in nodes:
            if n.i <= split_est:
                continue
            raw = lines[n.i]
            if n.op == P._ASM and n.i not in spin_lines:
                # classified with the hoist pass's exact allowlist, never by reading the text: assembly off it may do
                # anything; only quiet entries (timer / SM-id reads, the launch_dependents trigger - its wait is for
                # this grid's completion -, register-only ops, the stamp's plain store, which R3 attributes) order
                # nothing for this CTA's reads
                if P._asm_entry(raw) is None:
                    raise Fail("inline asm off the allowlist in the warp-specialized region")
                if _asm_class(raw) != "quiet":
                    raise Fail("side-effecting inline asm in the warp-specialized region")
            if n.op == P._EXTERN and not P._is_pure(raw):
                raise Fail("impure external function in the warp-specialized region")
            if n.op in ("tt.atomic_rmw", "tt.atomic_cas") and re.search(r"\b(acquire|acq_rel|seq_cst)\b", P._code(raw)):
                raise Fail("acquiring atomic in the warp-specialized region")
            if n.op == "tt.load" and "isVolatile = true" in raw:
                raise Fail("volatile load in the warp-specialized region")
            if n.op == "tt.call" and _callee(raw) in sync_funcs | ({_callee(raw)} - {g.name for g in funcs}):
                # the callee's acquires (e.g. a readiness spin outlined into a noinline helper) order the consumer's
                # reads, but only spins of the entry function are cloned into the activation producer
                raise Fail(f"call to a synchronizing function ({_callee(raw)}) in the warp-specialized region")

        # (rule E) a load the producers clone reads arguments no op of the region writes: every write to them is in the
        # prefix (this CTA: program order before the warp_specialize; other CTAs: ordered by the spins the producers
        # clone, exactly as the consumer's load was), so the clone and the consumer's load read the same bytes
        for i, X in dep_loads.items():
            if i <= split_est:
                raise Fail("producer cone loads in the ring-free prefix")
            for w in nodes:
                if w.i > split_est and write_hits(w) & X:
                    raise Fail(f"producer cone loads {sorted(X)}, written in the warp-specialized region")
        if dep_loads:
            clone_spins(sorted(dep_loads), w_spin_ops)
        act_copies = [c for c in copies if view_ring[dst_of(c)].name in act_names]
        # (R3) where the activation is written: never in the root that reads it; in an EARLIER root of the region only
        # through generic-proxy stores, and then the reading root's activation copies wait for a CTA-local ordering
        # point the consumer passes after that root (same-CTA program order restored; other CTAs' writes are ordered
        # by the cloned spins exactly as the consumer's reads were).  Writes before the region happen-before both
        # partitions (warp_specialize entry); writes in a LATER root only follow reads whose copies completed before
        # the consumer passed their stages (ready mbarrier = copy completion).
        async_funcs = _async_writing_functions(lines, funcs) if act_names else set()
        for rname in act_names:
            X = act_roots[rname]
            rc = [c for c in act_copies if view_ring[dst_of(c)].name == rname]
            tops = {top_of(c).i for c in rc}
            for w in nodes:
                if not write_hits(w) & X:
                    continue
                tw = top_of(w)
                if tw.i in tops:
                    raise Fail(f"activation {rname} written in the root that reads it")
                if w.i <= split_est:
                    continue   # written before the partitions exist
                if not (w.op in _GENERIC_WRITES or w.op == "tt.call" and _callee(lines[w.i]) in by_name
                        and _callee(lines[w.i]) not in async_funcs):
                    raise Fail(f"activation {rname} written by {w.op} in the warp-specialized region")
                for tr in tops:
                    if tr > tw.i and (tr not in order_after or order_after[tr].i < tw.i):
                        order_after[tr] = tw
        # (R1) clone every spin of the region that precedes an activation copy in program order (or shares a loop
        # with one): the activation producer then has performed every acquire the consumer had at each copy
        if act_copies:
            clone_spins([c.i for c in act_copies], spin_ops)
        # cloning spins may pull more dependent loads into the cones (their addresses): those need the same rules
        for i, X in dep_loads.items():
            if i <= split_est:
                raise Fail("producer cone loads in the ring-free prefix")
            for w in nodes:
                if w.i > split_est and write_hits(w) & X:
                    raise Fail(f"producer cone loads {sorted(X)}, written in the warp-specialized region")
        if dep_loads:
            clone_spins(sorted(dep_loads), w_spin_ops)

    # weight-producer lookahead (rule E): when W's dependent cones make it clone EVERY readiness spin the consumer passes
    # before the region's first stage, W issues nothing ahead of the consumer there (B5-9: the MoE MK's FC1 root, whose
    # route load needs the only spin before its tiles); the root-suffix policy does not start a ring at such a root.
    # Exception (R6, verify5 la.py): those spins and the first stage share a loop (W runs on into the loop's later
    # tasks while the consumer still computes this one) and every spin W clones in that loop polls state no op of the
    # kernel writes (published before the launch or by another grid): no progress of this kernel gates W there, so W
    # runs ahead by the ring's depth. A spin on state this kernel publishes (the MoE MK's tile-dependency counters,
    # static and dynamic alike) waits for an in-kernel producer on the critical path - W gets no earlier than the
    # consumer (B5-9: static FC2-on +4.5 / FC1-on +4.9 us vs the static hoist's +2.4).
    w_lookahead = True
    if dep_loads:
        first_stage = min(c.i for c in copies if preds[c.i] is not DEAD)
        pre = [g for g in spin_groups if split_est < g[0].i < first_stage]
        w_lookahead = not pre or any(not any(x.i in w_spin_ops for x in g) for g in pre)
        if not w_lookahead:
            w_lookahead = _runs_ahead_into_later_tasks(lines, by_line[first_stage], pre, spin_groups, w_spin_ops,
                                                        f.roots, read_only)
    # spin-gated (R6-8): the consumer passes a readiness spin before the ring's first stage that the weight producer
    # does not clone, so W streams the stages while the consumer waits (a cross-root prefetch). Such a ring may run
    # fewer CTAs per SM than its fallback, down to 2 (ttgir_weight_prefetch._Fallback): the fallback's extra CTAs would
    # spin as well.
    spin_gated = False
    if spin_groups and split_est is not None:
        first_stage = min(c.i for c in copies if preds[c.i] is not DEAD)
        spin_gated = any(split_est < g[0].i < first_stage and not any(x.i in w_spin_ops for x in g)
                         for g in spin_groups)

    # ---------------- shared byte rings (stage 2)
    k_of = {r.name: 1 for r in sel}
    byte_rings = []
    if group < 1 or D % group:
        raise Fail(f"ring depth {D} not a multiple of the release group {group}")
    NB = D // group
    bar_ty = f"!ttg.memdesc<{D}x1xi64, #rg_bar, #smem, mutable>"
    ebar_ty = f"!ttg.memdesc<{NB}x1xi64, #rg_bar, #smem, mutable>"
    bar1 = "!ttg.memdesc<1xi64, #rg_bar, #smem, mutable>"
    view_ty = {}
    if shared:
        role = {}
        for cs in pstages.values():
            for k, c in enumerate(sorted(cs, key=lambda c: (view_ring[dst_of(c)].name in act_names, c.i))):
                rn = view_ring[dst_of(c)].name
                if role.setdefault(rn, k) != k:
                    raise Fail("ring with two roles")
        groups = {}
        for r in sel:
            groups.setdefault(role[r.name], []).append(r)
        for g, rs in sorted(groups.items()):
            bmax = max(r.tile_bytes for r in rs)
            # Every member tile is reinterpreted at slot base s * bmax of ONE i8 allocation: the allocation must carry
            # the strictest alignment any member encoding assumes (128B-swizzled nvmma tiles: 1 KB), and so must
            # every slot base (Triton's allocator gives an i8 #rg_flat buffer only 16 B unless told otherwise).
            align = max(_enc_alignment(text, r.enc) for r in rs)
            if bmax % align:
                raise Fail(f"slot bytes {bmax} not a multiple of the member alignment {align}")
            name = f"%rg_bytes{g}"
            bt = f"!ttg.memdesc<{D}x{bmax}xi8, #rg_flat, #smem, mutable>"
            byte_rings.append((name, bt, bmax, rs, align))
            for r in rs:
                if bmax % r.tile_bytes:
                    raise Fail("tile bytes do not divide the slot")
                k_of[r.name] = bmax // r.tile_bytes
                view_ty[r.name] = f"!ttg.memdesc<{D * k_of[r.name]}x{'x'.join(map(str, r.tile))}x{r.elem}, {r.enc}>"
    else:
        for r in sel:
            view_ty[r.name] = f"!ttg.memdesc<{D}x{'x'.join(map(str, r.tile))}x{r.elem}, {r.enc}>"

    # ---------------- emit consumer (default region)
    edits, before, after = {}, {}, {}
    drop = {r.alloc.i for r in sel} | {d.i for r in sel for d in r.deallocs} | copy_set | bar_drop
    for r in sel:
        for ix in r.views:
            if all(u.i in copy_set for u in users.get(ix.results[0], [])):
                drop.add(ix.i)
    moved_tokens = {c.results[0] for c in copies if c.results}
    # consumer stages with a cp.async member: generic-proxy writes, made visible to MMA reads (async proxy) by a fence
    # after the ready wait; TMA-only stages need none (TMA writes and MMA reads are both async proxy)
    cp_rings = {r.name for r in sel if r.copies and r.copies[0].i not in tma_of}
    fence_of = {key: bool(ring_of_c[key] & cp_rings) for key in cinfo}
    uid = [0]

    def fresh(b):
        uid[0] += 1
        return f"%rg_{b}{uid[0]}"

    def view_line(ind, res, ring_name, slot_val, ring_ref=None):
        k = k_of[ring_name]
        out = []
        idx = slot_val
        if k != 1:
            kc, idx = fresh("k"), fresh("ks")
            out.append(f"{ind}{kc} = arith.constant {k} : i32")
            out.append(f"{ind}{idx} = arith.muli {slot_val}, {kc} : i32")
        r = next(x for x in sel if x.name == ring_name)
        out.append(f"{ind}{res} = ttg.memdesc_index {ring_ref or ring_name}[{idx}] : {view_ty[ring_name]} -> {r.slot_type()}")
        return out

    def add_result(n, k=1):
        """Give loop / if `n` k more results; returns (head line, first new result value, had results)."""
        code = lines[n.i]
        m = (_FOR if n.op == "scf.for" else _IF).match(P._code(code))
        res, cnt = m.group("res"), m.group("n")
        if res and not cnt:
            raise Fail("single-result region op")
        if res:
            nres = int(cnt)
            code = code.replace(f"{res}:{nres} =", f"{res}:{nres + k} =", 1)
            return code, f"{res}#{nres}", True
        newres = fresh("res")
        code = re.sub(r"^(\s*)scf\.", rf"\1{newres}{':' + str(k) if k > 1 else ''} = scf.", code, count=1)
        return code, (f"{newres}#0" if k > 1 else newres), False

    a0_of = {tk["A0"].i: w for w, tk in tickets.items()}
    a1_of = {tk["A1"].i: w for w, tk in tickets.items()}
    jinit, jdo, jout = {}, {}, {}
    MB = 4
    mb_ty = f"!ttg.memdesc<{MB}x1xi32, #rg_flat, #smem, mutable>"
    mb1 = "!ttg.memdesc<1xi32, #rg_flat, #smem, mutable>"
    mbar_ty = f"!ttg.memdesc<{MB}x1xi64, #rg_bar, #smem, mutable>"

    def mbox_get(ind, j, res, one="%rg_one", mk=None):
        mk = mk or fresh
        s_, q_, ph_, fb, mv, tv, eb, jn, cM = (mk(x) for x in ("ms", "mq", "mph", "mfb", "mv", "mt", "meb", "mj", "cM"))
        ra, rb_, rm = mk("ra"), mk("rb"), mk("rm")
        return [f"{ind}{cM} = arith.constant {MB} : i32",
                f"{ind}{s_} = arith.remsi {j}, {cM} : i32",
                f"{ind}{q_} = arith.divsi {j}, {cM} : i32",
                f"{ind}{ph_} = arith.andi {q_}, {one} : i32",
                f"{ind}{fb} = ttg.memdesc_index %rg_mfull[{s_}] : {mbar_ty} -> {bar1}",
                f"{ind}ttng.wait_barrier {fb}, {ph_} : {bar1}",
                f"{ind}{mv} = ttg.memdesc_index %rg_mbox[{s_}] : {mb_ty} -> {mb1}",
                f"{ind}{tv} = ttg.local_load {mv} : {mb1} -> tensor<1xi32, #rg_blk1>",
                f'{ind}{res} = "tt.reduce"({tv}) <{{axis = 0 : i32}}> ({{',
                f"{ind}^bb0({ra}: i32, {rb_}: i32):",
                f"{ind}  {rm} = arith.maxsi {ra}, {rb_} : i32",
                f"{ind}  tt.reduce.return {rm} : i32",
                f"{ind}}}) : (tensor<1xi32, #rg_blk1>) -> i32",
                f"{ind}{eb} = ttg.memdesc_index %rg_mempty[{s_}] : {mbar_ty} -> {bar1}",
                f"{ind}ttng.arrive_barrier {eb}, 1 : {bar1}",
                f"{ind}{jn} = arith.addi {j}, {one} : i32"], jn

    def while_edit(n, cinit, jin, bc, bj, emit_to=None):
        """Add carried (counter, mailbox counter) to a ticket while; returns (head, cond, bb0 lines, result base)."""
        m = _WHILE.match(P._code(lines[n.i]))
        res, cnt = m.group("res"), m.group("n")
        nold = int(cnt) if cnt else (1 if res else 0)
        if res and not cnt and len(users.get(res, [])) > 0:
            raise Fail("single-result while used after the loop")
        wc, wj = fresh("wc"), fresh("wj")
        rty = m.group("rty").strip()
        rlist = rty[1:-1] if rty.startswith("(") else rty
        newres = res or fresh("wres")
        head = (f"{m.group('ind')}{newres}:{nold + 2} = scf.while ({m.group('args')}, {wc} = {cinit}, {wj} = {jin}) : "
                f"({m.group('ty')}, i32, i32) -> ({rlist}, i32, i32) {{")
        cond = n.regions[0][-1]
        cc = P._code(lines[cond.i]).strip()
        vals, _, tys = cc.partition(" : ")
        cl = " " * P._indent(lines[cond.i]) + f"{vals}, {wc}, {wj} : {tys}, i32, i32"
        bb = lines[n.seps[0] + 1]
        tys_after = [t.strip() for t in m.group("ty").split(",")]
        tk = tickets[n.i]
        bbl = " " * P._indent(bb) + "^bb0(" + ", ".join(f"{a}: {t}" for a, t in zip(tk["after"], tys_after)) + \
            f", {bc}: i32, {bj}: i32):"
        return head, cl, bbl, f"{newres}#{nold}"

    pfctx = {}
    rename = {}

    def reader_type(rd):
        mm = re.search(r"->\s*(tensor<.*>)\s*$", P._code(lines[rd.i]).rstrip())
        if mm is None:
            raise Fail("reader type")
        return mm.group(1)

    def prefetch_key(n):
        """The single local_load stage of constant-bound loop `n` eligible for register prefetch, or None."""
        if not prefetch or D < 2 * group:
            return None
        keys = [k for k, (ixs, rds, kind) in cinfo.items() if any(a is n for a in _ancestors(ixs[0]))]
        if len(keys) != 1:
            return None
        ixs, rds, kind = cinfo[keys[0]]
        if kind != "ttg.local_load" or ixs[0].parent is not n:
            return None
        mm = _FOR.match(P._code(lines[n.i]))
        lb, ub, st = (const_of(mm.group(g)) for g in ("lb", "ub", "st"))
        if None in (lb, ub, st) or st <= 0 or ub <= lb:
            return None
        for ix in ixs:   # chain view -> [memdesc_trans] -> local_load only, all in this body
            for u in users.get(ix.results[0], []):
                if u.parent is not n:
                    return None
                if u.op == "ttg.memdesc_trans":
                    if any(w.op != "ttg.local_load" or w.parent is not n for w in users.get(u.results[0], [])):
                        return None
                elif u.op != "ttg.local_load":
                    return None
        return keys[0]

    def chain_lines(ixs, ind, slot, suffix, drop_old):
        """Clone view -> [trans] -> local_load chains of a stage for `slot`; returns (lines, {old reader: new})."""
        out, newname = [], {}
        for ix in ixs:
            nv = fresh("pv")
            out += view_line(ind, nv, ix.uses[0], slot)
            if drop_old:
                drop.add(ix.i)
            for u in users.get(ix.results[0], []):
                srcs = [(u, nv)]
                if u.op == "ttg.memdesc_trans":
                    nt = fresh("pt")
                    out.append(ind + re.sub(rf"^{re.escape(u.results[0])}\s*=", f"{nt} =",
                                            P._code(lines[u.i]).strip().replace(ix.results[0], nv, 1)))
                    if drop_old:
                        drop.add(u.i)
                    srcs = [(w, nt) for w in users.get(u.results[0], [])]
                    src_name = u.results[0]
                else:
                    src_name = ix.results[0]
                for rd, src in srcs:
                    nl = fresh("pl" + suffix)
                    code = P._code(lines[rd.i]).strip()
                    code = re.sub(r"\s+token\s+%[\w$.#-]+", "", code)
                    code = re.sub(rf"^{re.escape(rd.results[0])}\s*=", f"{nl} =", code)
                    code = code.replace(f"ttg.local_load {src_name}", f"ttg.local_load {src}", 1)
                    out.append(ind + code)
                    newname[rd.results[0]] = nl
                    if drop_old:
                        drop.add(rd.i)
        return out, newname

    def flow(block, cur):
        for n in block:
            if n.i in a0_of:
                ind = " " * P._indent(lines[n.i])
                gl, jn = mbox_get(ind, "%rg_c0", n.results[0])
                edits[n.i] = gl
                jinit[a0_of[n.i]] = jn
                continue
            if n.i in a1_of:
                ind = " " * P._indent(lines[n.i])
                gl, jn = mbox_get(ind, jdo[a1_of[n.i]], n.results[0])
                edits[n.i] = gl
                jout[a1_of[n.i]] = jn
                continue
            if n.op == "scf.while" and n.i in tickets:
                bc, bj = fresh("bc"), fresh("bj")
                head, cl, bbl, outbase = while_edit(n, cur, jinit[n.i], bc, bj)
                tk = tickets[n.i]
                edits[n.i] = [head]
                edits[tk["cond"].i] = [cl]
                edits[n.seps[0] + 1] = [bbl]
                jdo[n.i] = bj
                inner = flow(n.regions[1], bc)
                yc = P._code(lines[tk["yld"].i]).strip()[len("scf.yield"):]
                vals, _, tys = yc.partition(":")
                edits[tk["yld"].i] = [" " * P._indent(lines[tk["yld"].i]) +
                                      f"scf.yield {vals.strip()}, {inner}, {jout[n.i]} : {tys.strip()}, i32, i32"]
                cur = outbase
                continue
            if n.op == "scf.while" and n.i in c_ctr:
                raise Fail("stage inside a non-ticket while")
            if n.i in c_ctr and n.op == "scf.for":
                m = _FOR.match(P._code(lines[n.i]))
                arg = fresh("cnt")
                pk = prefetch_key(n)
                extra_args, extra_tys = [], []
                if pk is not None:
                    ixs_, rds_, _ = cinfo[pk]
                    ind_ = " " * P._indent(lines[n.i])
                    s0, q0, ph0, rb0, cD0 = (fresh(x) for x in ("s0", "q0", "ph0", "rb0", "cD0"))
                    pro = [f"{ind_}{cD0} = arith.constant {D} : i32", f"{ind_}{s0} = arith.remui {cur}, {cD0} : i32",
                           f"{ind_}{q0} = arith.divui {cur}, {cD0} : i32", f"{ind_}{ph0} = arith.andi {q0}, %rg_one : i32",
                           f"{ind_}{rb0} = ttg.memdesc_index %rg_ready[{s0}] : {bar_ty} -> {bar1}",
                           f"{ind_}ttng.wait_barrier {rb0}, {ph0} : {bar1}"]
                    if neg == "early":
                        eb0 = fresh("eb0")
                        pro += [f"{ind_}{eb0} = ttg.memdesc_index %rg_empty[{s0}] : {ebar_ty} -> {bar1}",
                                f"{ind_}ttng.arrive_barrier {eb0}, 1 : {bar1}"]
                    cl, first = chain_lines(ixs_, ind_, s0, "0", False)
                    before.setdefault(n.i, []).extend(pro + cl)
                    order = [rd.results[0] for rd in sorted(rds_, key=lambda x: x.i)]
                    pargs = {o: fresh("pfa") for o in order}
                    extra_args = [f"{pargs[o]} = {first[o]}" for o in order]
                    extra_tys = [reader_type(next(rd for rd in rds_ if rd.results[0] == o)) for o in order]
                    rename.update({o: pargs[o] for o in order})
                    pfctx[pk] = dict(loop=n, order=order, iv=m.group("iv"), ub=m.group("ub"), st=m.group("st"))
                head, outv, had = add_result(n, 1 + len(extra_args))
                ia_new = ", ".join([f"{arg} = {cur}"] + extra_args)
                ty_new = ", ".join(["i32"] + extra_tys)
                if m.group("ia") is None:
                    head = re.sub(r"(step\s+%[\w$.#-]+)\s*:", rf"\1 iter_args({ia_new}) -> ({ty_new}) :", head, count=1)
                else:
                    head = head.replace(f"iter_args({m.group('ia')})", f"iter_args({m.group('ia')}, {ia_new})", 1)
                    head = head.replace(f"-> ({m.group('ty')})", f"-> ({m.group('ty')}, {ty_new})", 1)
                    if not had:  # loop with iter_args but no results bound: never printed that way
                        raise Fail("loop with unbound results")
                edits[n.i] = [head]
                inner = flow(n.regions[0], arg)
                if pk is not None:
                    _yield(n, 0, inner, [(pfctx[pk]["next"][o], t) for o, t in zip(pfctx[pk]["order"], extra_tys)])
                else:
                    _yield(n, 0, inner)
                cur = outv
            elif n.i in c_ctr and n.op == "scf.if":
                m = _IF.match(P._code(lines[n.i]))
                head, outv, had = add_result(n)
                if m.group("ty"):
                    head = head.replace(f"-> ({m.group('ty')})", f"-> ({m.group('ty')}, i32)", 1)
                else:
                    head = re.sub(rf"(scf\.if\s+{V})\s*\{{", r"\1 -> (i32) {", head, count=1)
                edits[n.i] = [head]
                for k, reg in enumerate(n.regions):
                    _yield(n, k, flow(reg, cur))
                if len(n.regions) == 1:
                    ind = " " * (P._indent(lines[n.i]) + 2)
                    before.setdefault(n.end, []).extend(
                        [" " * P._indent(lines[n.i]) + "} else {", f"{ind}scf.yield {cur} : i32"])
                cur = outv
            for key, (ixs, readers, kind) in cinfo.items():
                if ixs[0].i != n.i:
                    continue
                ind = " " * P._indent(lines[n.i])
                if key in pfctx:
                    cx = pfctx[key]
                    L_ = cx["loop"]
                    s, q, ph, nxt, cD, sn, qn, phn, ivn, hn, srd, rbn = (fresh(x) for x in (
                        "s", "q", "ph", "cn", "cD", "sn", "qn", "phn", "ivn", "hn", "srd", "rbn"))
                    before.setdefault(n.i, []).extend([
                        f"{ind}{cD} = arith.constant {D} : i32",
                        f"{ind}{s} = arith.remui {cur}, {cD} : i32",
                        f"{ind}{nxt} = arith.addi {cur}, %rg_one : i32",
                        f"{ind}{sn} = arith.remui {nxt}, {cD} : i32",
                        f"{ind}{qn} = arith.divui {nxt}, {cD} : i32",
                        f"{ind}{phn} = arith.andi {qn}, %rg_one : i32",
                        f"{ind}{ivn} = arith.addi {cx['iv']}, {cx['st']} : i32",
                        f"{ind}{hn} = arith.cmpi slt, {ivn}, {cx['ub']} : i32",
                        f"{ind}{srd} = arith.select {hn}, {sn}, {s} : i32",
                        f"{ind}{rbn} = ttg.memdesc_index %rg_ready[{sn}] : {bar_ty} -> {bar1}",
                        f"{ind}ttng.wait_barrier {rbn}, {phn}, {hn} : {bar1}"])
                    if neg == "early":
                        ebn = fresh("ebn")
                        before[n.i].extend([f"{ind}scf.if {hn} {{",
                                            f"{ind}  {ebn} = ttg.memdesc_index %rg_empty[{sn}] : {ebar_ty} -> {bar1}",
                                            f"{ind}  ttng.arrive_barrier {ebn}, 1 : {bar1}", f"{ind}}}"])
                    cl, newn = chain_lines(ixs, ind, srd, "n", True)
                    cx["next"] = newn
                    tail = list(cl)
                    if neg == "early":
                        pass
                    elif group == 1:
                        eb = fresh("eb")
                        tail += [f"{ind}{eb} = ttg.memdesc_index %rg_empty[{s}] : {ebar_ty} -> {bar1}",
                                 f"{ind}ttng.arrive_barrier {eb}, 1 : {bar1}"]
                    else:
                        cG, cG1, gm, isl, gq, gb, cNB, eb = (fresh(x) for x in ("cG", "cG1", "gm", "isl", "gq", "gb", "cNB", "eb"))
                        tail += [f"{ind}{cG} = arith.constant {group} : i32", f"{ind}{cG1} = arith.constant {group - 1} : i32",
                                 f"{ind}{cNB} = arith.constant {NB} : i32", f"{ind}{gm} = arith.remui {cur}, {cG} : i32",
                                 f"{ind}{isl} = arith.cmpi eq, {gm}, {cG1} : i32", f"{ind}scf.if {isl} {{",
                                 f"{ind}  {gq} = arith.divui {cur}, {cG} : i32", f"{ind}  {gb} = arith.remui {gq}, {cNB} : i32",
                                 f"{ind}  {eb} = ttg.memdesc_index %rg_empty[{gb}] : {ebar_ty} -> {bar1}",
                                 f"{ind}  ttng.arrive_barrier {eb}, 1 : {bar1}", f"{ind}}}"]
                    last = L_.regions[0][-1] if L_.regions[0] else None
                    at = last.i if last is not None and last.op == "scf.yield" else L_.seps[0]
                    before.setdefault(at, []).extend(tail)
                    cur = nxt
                    continue
                s, q, ph, rb, eb, nxt, cD = (fresh(x) for x in ("s", "q", "ph", "rb", "eb", "cn", "cD"))
                before.setdefault(n.i, []).extend([
                    f"{ind}{cD} = arith.constant {D} : i32",
                    f"{ind}{s} = arith.remui {cur}, {cD} : i32",
                    f"{ind}{q} = arith.divui {cur}, {cD} : i32",
                    f"{ind}{ph} = arith.andi {q}, %rg_one : i32",
                    f"{ind}{nxt} = arith.addi {cur}, %rg_one : i32"])
                for ix in ixs:
                    edits[ix.i] = view_line(ind, ix.results[0], ix.uses[0], s)
                wl = [f"{ind}{rb} = ttg.memdesc_index %rg_ready[{s}] : {bar_ty} -> {bar1}",
                      f"{ind}ttng.wait_barrier {rb}, {ph} : {bar1}"]
                if kind == "ttng.tc_gen5_mma" and fence_of[key]:
                    wl.append(f"{ind}ttng.fence_async_shared {{bCluster = false}}")
                before.setdefault(readers[0].i, []).extend(wl)
                if neg == "early":
                    before[readers[0].i].extend([f"{ind}{eb} = ttg.memdesc_index %rg_empty[{s}] : {ebar_ty} -> {bar1}",
                                                 (f"{ind}ttng.arrive_barrier {eb}, 1 : {bar1}" if kind == "ttg.local_load"
                                                  else f"{ind}ttng.tc_gen5_commit {eb} : {bar1}")])
                    if any(b.i in c_ctr and ixs[0].i < b.i < readers[-1].i for b in block):
                        raise Fail("region op between a stage's first read and its release")
                    cur = nxt
                    continue
                if group == 1:
                    rel = [f"{ind}{eb} = ttg.memdesc_index %rg_empty[{s}] : {ebar_ty} -> {bar1}",
                           (f"{ind}ttng.arrive_barrier {eb}, 1 : {bar1}" if kind == "ttg.local_load"
                            else f"{ind}ttng.tc_gen5_commit {eb} : {bar1}")]
                else:
                    # release the G-stage group after reading its last stage (count % G == G - 1)
                    cG, cG1, gm, isl, gq, gb = (fresh(x) for x in ("cG", "cG1", "gm", "isl", "gq", "gb"))
                    cNB = fresh("cNB")
                    rel = [f"{ind}{cG} = arith.constant {group} : i32", f"{ind}{cG1} = arith.constant {group - 1} : i32",
                           f"{ind}{cNB} = arith.constant {NB} : i32",
                           f"{ind}{gm} = arith.remui {cur}, {cG} : i32",
                           f"{ind}{isl} = arith.cmpi eq, {gm}, {cG1} : i32",
                           f"{ind}scf.if {isl} {{",
                           f"{ind}  {gq} = arith.divui {cur}, {cG} : i32",
                           f"{ind}  {gb} = arith.remui {gq}, {cNB} : i32",
                           f"{ind}  {eb} = ttg.memdesc_index %rg_empty[{gb}] : {ebar_ty} -> {bar1}",
                           (f"{ind}  ttng.arrive_barrier {eb}, 1 : {bar1}" if kind == "ttg.local_load"
                            else f"{ind}  ttng.tc_gen5_commit {eb} : {bar1}"),
                           f"{ind}}}"]
                after.setdefault(readers[-1].i, []).extend(rel)
                if any(b.i in c_ctr and ixs[0].i < b.i < readers[-1].i for b in block):
                    raise Fail("region op between a stage's first read and its release")
                cur = nxt
        return cur

    def _yield(n, k, val, extra=()):
        reg = n.regions[k]
        last = reg[-1] if reg else None
        yind = " " * (P._indent(lines[n.i]) + 2)
        ev = "".join(f", {v}" for v, _ in extra)
        et = "".join(f", {t}" for _, t in extra)
        if last is not None and last.op == "scf.yield":
            code = P._code(lines[last.i]).strip()[len("scf.yield"):]
            vals, _, tys = code.partition(":")
            edits[last.i] = [f"{yind}scf.yield {vals.strip()}, {val}{ev} : {tys.strip()}, i32{et}"]
        else:
            before.setdefault(n.seps[k], []).append(f"{yind}scf.yield {val}{ev} : i32{et}")

    flow(body, "%rg_c0")
    wo_nodes = sorted({tw.i: tw for tw in order_after.values()}.values(), key=lambda x: x.i)
    wo_index = {tw.i: k for k, tw in enumerate(wo_nodes)}
    wo_ty = f"!ttg.memdesc<{max(1, len(wo_nodes))}x1xi64, #rg_bar, #smem, mutable>"
    for k, tw in enumerate(wo_nodes):
        # all consumer threads' stores of root tw precede the arrive (arrive_barrier = bar.sync of the default
        # partition + one elected release arrive at CTA scope); the activation producer acquires it before reading
        ind = " " * P._indent(lines[tw.i])
        after.setdefault(tw.end, []).extend([f"{ind}%rg_wov{k} = ttg.memdesc_index %rg_wo[%rg_woc{k}] : {wo_ty} -> {bar1}",
                                              f"{ind}ttng.arrive_barrier %rg_wov{k}, 1 : {bar1}"])
    if drop_waits:
        # A consumer ttg.async_wait whose every user is a read of a moved slot (now guarded by its ready mbarrier),
        # with no other shared-memory reader before the next wait in its block, protects nothing any more: its
        # cp.async.wait_group + bar.sync is vestigial.
        moved_readers = {rd.i for _, rds, _ in cinfo.values() for rd in rds}
        slot_views = {ix.results[0] for r in sel for ix in r.views}
        for n in nodes:
            if n.op != "ttg.async_wait" or not n.results:
                continue
            us = users.get(n.results[0], [])
            if not us or any(u.i not in moved_readers or u.op != "ttg.local_load" for u in us):
                continue
            blk = siblings_of(n)
            k0 = blk.index(n)
            ok = True
            for m_ in blk[k0 + 1:]:
                if m_.op == "ttg.async_wait":
                    break
                if m_.regions or (m_.op in ("ttg.local_load", "ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled")
                                  and m_.i not in moved_readers):
                    ok = False
                    break
            if not ok:
                continue
            drop.add(n.i)
            for u in us:
                code = edits.get(u.i, [lines[u.i]])
                if len(code) != 1:
                    raise Fail("edited reader")
                edits[u.i] = [re.sub(rf"\s+token\s+{re.escape(n.results[0])}(?![\w$.-])", "", code[0])]
    for n in nodes:
        if n.op == "ttg.async_commit_group" and set(n.uses) & moved_tokens:
            kept = [u for u in n.uses if u not in moved_tokens]
            ind = " " * P._indent(lines[n.i])
            edits[n.i] = [f"{ind}{n.results[0]} = ttg.async_commit_group" + (f" tokens {', '.join(kept)}" if kept else "")]
    if act_names and not any(n.op == P._COPY and n.i not in copy_set and n.i > split_est for n in nodes):
        # the consumer issues no cp.async any more: its ttg.async_wait ops (and the bar.sync Triton's membar analysis
        # puts after them) order nothing; the moved slots are ordered by their ready mbarriers.  Its (now empty)
        # commit groups become poison tokens (only dropped waits and loop-carried token slots consume them).
        for n in nodes:
            if n.op == "ttg.async_commit_group" and n.i > split_est and n.results:
                edits[n.i] = [" " * P._indent(lines[n.i]) + f"{n.results[0]} = ub.poison : !ttg.async.token"]
            if n.op != "ttg.async_wait" or n.i <= split_est:
                continue
            us = users.get(n.results[0], []) if n.results else []
            if any(u.op != "ttg.local_load" for u in us):
                continue
            drop.add(n.i)
            for u in us:
                code = edits.get(u.i, [lines[u.i]])
                if len(code) != 1:
                    raise Fail("edited reader")
                edits[u.i] = [re.sub(rf"\s+token\s+{re.escape(n.results[0])}(?![\w$.-])", "", code[0])]
    if act_names or dep_loads:
        # the region R1-R3 (and rule E) reasoned about IS the emitted one: the warp_specialize starts at split_est (a
        # ring edit in the prefix would fail the "ring code before the warp_specialize" check below)
        split_at = split_est
    elif late_ws:
        touched = set(edits) | set(before) | set(after) | copy_set | {ix.i for r in sel for ix in r.views}
        touched -= {r.alloc.i for r in sel} | {d.i for r in sel for d in r.deallocs}
        for n in body:
            if n.i > split_at and any(n.i <= k <= n.end for k in touched):
                split_at = n.i - 1
                break
    for v in pcaps:
        if not f.start < by_res[v].i <= split_at:
            raise Fail(f"captured {v} not defined before the warp_specialize")
    ind0 = " " * (P._indent(lines[f.start]) + 2)
    default = [f"{ind0}  %rg_one = arith.constant 1 : i32"]
    for k in range(split_at + 1, f.end - 1):
        if P._code(lines[k]).strip().startswith("tt.return"):
            if P._indent(lines[k]) != len(ind0):
                raise Fail("nested return")
            continue
        default.extend(before.get(k, []))
        if k not in drop:
            got = edits.get(k, [lines[k]])
            if rename:
                got = [_rename(l, rename) for l in got]
            default.extend(got)
        default.extend(after.get(k, []))
    default.extend(before.get(f.end - 1, []))

    if unroll or unroll_factor:
        consts = {}
        for l in lines:
            m_ = re.match(rf"^\s*({V}) = arith\.constant (-?\d+) : i(?:32|64)\b", P._code(l))
            if m_:
                consts[m_.group(1)] = int(m_.group(2))
        default = _unroll_loops(default, consts, unroll, marker="%rg_rb", factor=unroll_factor)

    # ---------------- emit producer (partition0)
    rolled_copies = {c for _lb, _ub, pro in roll_loops.values() for c in pro}
    head = P._code(lines[f.start])
    ptypes = {m.group(1): m.group(2) for m in re.finditer(r"(%[\w$.-]+)\s*:\s*(![\w.]+<[^{}]*?>(?=[\s,){])|i\d+|f\d+|bf16)", head.split("->")[0])}
    ring_caps = [b[0] for b in byte_rings] if shared else [r.name for r in sel]
    ring_cap_ty = [b[1] for b in byte_rings] if shared else [view_ty[r.name] for r in sel]
    mb_caps = ["%rg_mbox", "%rg_mfull", "%rg_mempty"] if tickets else []
    mb_cap_ty = [mb_ty, mbar_ty, mbar_ty] if tickets else []
    wo_caps = (["%rg_wo"] + [f"%rg_woc{k}" for k in range(len(wo_nodes))]) if wo_nodes else []
    wo_cap_ty = ([wo_ty] + ["i32"] * len(wo_nodes)) if wo_nodes else []
    cap_all = ring_caps + ["%rg_ready", "%rg_empty"] + mb_caps + wo_caps + captures + pcaps
    cap_ty = ring_cap_ty + [bar_ty, ebar_ty] + mb_cap_ty + wo_cap_ty + [ptypes[c] for c in captures] + \
        [_scalar_type(lines[by_res[v].i]) for v in pcaps]
    pa = {v: f"%pa{k}" for k, v in enumerate(cap_all)}
    # the weight producer issues a cp.async member (its noinc arrivals count in the ready barrier) or TMA only
    w_has_cp = any(c.i not in tma_of and view_ring[dst_of(c)].name not in act_names for c in copies)

    def build(kind):
        """Producer partition body: kind "w" issues the weight copies of every stage, kind "a" (act mode) the
        activation copies behind the cloned spins; both wait / arrive on every stage of the global sequence."""
        ren = dict(pa)
        for k in keep:
            for r in P._results(lines[k]):
                ren.setdefault(r, "%p" + r[1:])
            m = _FOR.match(P._code(lines[k]))
            if m:
                ren.setdefault(m.group("iv"), "%p" + m.group("iv")[1:])
            if k in tickets:
                tk = tickets[k]
                for a in (tk["before"][tk["k"]][0], tk["after"][tk["k"]]):
                    ren.setdefault(a, "%p" + a[1:])

        def rn(s):
            return re.sub(V, lambda m: ren.get(m.group(0).split("#")[0], m.group(0).split("#")[0]) +
                          ("#" + m.group(0).split("#")[1] if "#" in m.group(0) else ""), s)

        def rn_raw(line):
            """rn() of a line without its location, string literals (inline asm text, constraints) kept verbatim."""
            raw = line[:len(P._code(line))]
            out, at = [], 0
            for m in P._STRING.finditer(raw):
                out += [rn(raw[at:m.start()]), m.group(0)]
                at = m.end()
            return "".join(out + [rn(raw[at:])])
        pind = ind0 + "  "
        prod = [f"{pind}%pr_one = arith.constant 1 : i32", f"{pind}%pr_c0 = arith.constant 0 : i32",
                f"{pind}%pr_D = arith.constant {D} : i32", f"{pind}%pr_G = arith.constant {group} : i32",
                f"{pind}%pr_NB = arith.constant {NB} : i32"]
        if ring_has_tma:
            prod.append(f"{pind}%pr_true = arith.constant true")
        ring_ref = {}
        if shared:
            for (name, bt, bmax, rs, _align) in byte_rings:
                for r in rs:
                    v = f"%pr_view_{r.name[1:]}"
                    prod.append(f"{pind}{v} = ttg.memdesc_reinterpret {pa[name]} : {bt} -> {view_ty[r.name]}")
                    ring_ref[r.name] = v
        else:
            for r in sel:
                ring_ref[r.name] = pa[r.name]
        pc = [0]

        def pf(b):
            pc[0] += 1
            return f"%pr_{b}{pc[0]}"

        pj_init, pj_do, pj_out = {}, {}, {}

        def mbox_put(ind, j, tval):
            s_, q_, ph_, pi_, eb, tv, mv, fb, jn, cM = (pf(x) for x in ("ms", "mq", "mph", "mpi", "meb", "mt", "mv", "mfb", "mj", "cM"))
            return [f"{ind}{cM} = arith.constant {MB} : i32",
                    f"{ind}{s_} = arith.remsi {j}, {cM} : i32",
                    f"{ind}{q_} = arith.divsi {j}, {cM} : i32",
                    f"{ind}{ph_} = arith.andi {q_}, %pr_one : i32",
                    f"{ind}{pi_} = arith.xori {ph_}, %pr_one : i32",
                    f"{ind}{eb} = ttg.memdesc_index {pa['%rg_mempty']}[{s_}] : {mbar_ty} -> {bar1}",
                    f"{ind}ttng.wait_barrier {eb}, {pi_} : {bar1}",
                    f"{ind}{tv} = tt.splat {tval} : i32 -> tensor<1xi32, #rg_blk1>",
                    f"{ind}{mv} = ttg.memdesc_index {pa['%rg_mbox']}[{s_}] : {mb_ty} -> {mb1}",
                    f"{ind}ttg.local_store {tv}, {mv} : tensor<1xi32, #rg_blk1> -> {mb1}",
                    f"{ind}{fb} = ttg.memdesc_index {pa['%rg_mfull']}[{s_}] : {mbar_ty} -> {bar1}",
                    f"{ind}ttng.arrive_barrier {fb}, 1 : {bar1}",
                    f"{ind}{jn} = arith.addi {j}, %pr_one : i32"], jn

        def emit_p(block, cur, ind):
            for n in block:
                if n.i not in keep and n.i not in copy_set:
                    continue
                if kind == "a" and block is body and n.i in order_after and neg != "noorder":
                    k = wo_index[order_after[n.i].i]
                    wv = pf("wo")
                    prod.extend([f"{ind}{wv} = ttg.memdesc_index {pa['%rg_wo']}[{pa['%rg_woc' + str(k)]}] : {wo_ty} -> {bar1}",
                                 f"{ind}ttng.wait_barrier {wv}, %pr_c0 : {bar1}"])
                code = P._code(lines[n.i])
                if n.i in spin_ops or n.i in w_spin_ops:
                    # rule E spins guard loads both producers clone; act spins guard the activation copies (A only)
                    if neg == "nowspin" and kind == "w" and n.i in w_spin_ops:
                        continue   # TEST ONLY: the weight producer loads its dependent addresses unordered
                    if not (n.i in w_spin_ops or (kind == "a" and neg != "nospin")):
                        continue
                    base = P._indent(lines[n.i])
                    for k in range(n.i, n.end + 1):
                        cl = rn_raw(lines[k]).strip()
                        prod.append(ind + " " * (P._indent(lines[k]) - base) + cl)
                    continue
                if n.i in a1_of and claim_early:
                    continue  # emitted at the top of the do-region (claim before this task's copies)
                if n.i in a0_of or n.i in a1_of:
                    w = a0_of.get(n.i, a1_of.get(n.i))
                    prod.append(ind + rn(code.strip()))
                    j = "%pr_c0" if n.i in a0_of else pj_do[w]
                    pl, jn = mbox_put(ind, j, rn(n.results[0]))
                    prod.extend(pl)
                    (pj_init if n.i in a0_of else pj_out)[w] = jn
                    continue
                if n.op == "scf.while":
                    tk = tickets[n.i]
                    k = tk["k"]
                    m = _WHILE.match(code)
                    ty = [t.strip() for t in m.group("ty").split(",")][k]
                    res, pc, pjj, bc, bj = pf("wr"), pf("wc"), pf("wj"), pf("bc"), pf("bj")
                    ta, tb = rn(tk["before"][k][0]), rn(tk["after"][k])
                    prod.append(f"{ind}{res}:3 = scf.while ({ta} = {rn(tk['before'][k][1])}, {pc} = {cur}, {pjj} = {pj_init[n.i]}) : "
                                f"({ty}, i32, i32) -> ({ty}, i32, i32) {{")
                    emit_p(n.regions[0], cur, ind + "  ")
                    prod.append(f"{ind}  scf.condition({rn(tk['cond'].uses[0])}) {ta}, {pc}, {pjj} : {ty}, i32, i32")
                    prod.append(f"{ind}}} do {{")
                    prod.append(f"{ind}^bb0({tb}: {ty}, {bc}: i32, {bj}: i32):")
                    pj_do[n.i] = bj
                    if claim_early:
                        a1 = tk["A1"]
                        prod.append(ind + "  " + rn(P._code(lines[a1.i]).strip()))
                        pl, jn = mbox_put(ind + "  ", bj, rn(a1.results[0]))
                        prod.extend(pl)
                        pj_out[n.i] = jn
                    inner = emit_p(n.regions[1], bc, ind + "  ")
                    prod.append(f"{ind}  scf.yield {rn(tk['A1'].results[0])}, {inner}, {pj_out[n.i]} : {ty}, i32, i32")
                    prod.append(f"{ind}}}")
                    cur = f"{res}#1"
                    continue
                if n.op == "scf.for":
                    m = _FOR.match(code)
                    arg, res = pf("c"), pf("r")
                    lbv, ubv = rn(m.group('lb')), rn(m.group('ub'))
                    if n.i in roll_loops:   # rolled producer: the loop issues the peeled prologue tiles itself
                        lbv, ubv = pf("rlb"), pf("rub")
                        prod.append(f"{ind}{lbv} = arith.constant {roll_loops[n.i][0]} : {m.group('ivt')}")
                        prod.append(f"{ind}{ubv} = arith.constant {roll_loops[n.i][1]} : {m.group('ivt')}")
                    prod.append(f"{ind}{res} = scf.for {rn(m.group('iv'))} = {lbv} to {ubv} step "
                                f"{rn(m.group('st'))} iter_args({arg} = {cur}) -> (i32) : {m.group('ivt')} {{")
                    prod.append(f"{ind}  scf.yield {emit_p(n.regions[0], arg, ind + '  ')} : i32")
                    prod.append(f"{ind}}}")
                    cur = res
                elif n.op == "scf.if":
                    m = _IF.match(code)
                    res = pf("r")
                    prod.append(f"{ind}{res} = scf.if {rn(m.group('c'))} -> (i32) {{")
                    prod.append(f"{ind}  scf.yield {emit_p(n.regions[0], cur, ind + '  ')} : i32")
                    prod.append(f"{ind}}} else {{")
                    if len(n.regions) > 1:
                        prod.append(f"{ind}  scf.yield {emit_p(n.regions[1], cur, ind + '  ')} : i32")
                    else:
                        prod.append(f"{ind}  scf.yield {cur} : i32")
                    prod.append(f"{ind}}}")
                    cur = res
                elif n.regions and _generic_op(lines[n.i]) in ("tt.reduce", "tt.scan"):
                    # rule E: an integer reduction / scan of a dependent cone, cloned verbatim (combiner included)
                    base = P._indent(lines[n.i])
                    for k in range(n.i, n.end + 1):
                        prod.append(ind + " " * (P._indent(lines[k]) - base) + rn_raw(lines[k]).strip())
                elif n.i in copy_set:
                    if preds[n.i] is DEAD or n.i in rolled_copies:
                        continue
                    key = (id(n.parent), slot_of_copy(n))
                    cs = sorted(pstages[key], key=lambda c: c.i)
                    if n.i != cs[-1].i:
                        continue
                    p = preds[n.i]
                    if p is DEAD:
                        continue
                    ind2 = ind + ("  " if p else "")
                    s, q, ph, pi, eb, rb = (pf(x) for x in ("s", "q", "ph", "pi", "eb", "rb"))
                    bl = [f"{ind2}{s} = arith.remui {cur}, %pr_D : i32",
                          f"{ind2}{q} = arith.divui {cur}, %pr_D : i32",
                          f"{ind2}{ph} = arith.andi {q}, %pr_one : i32",
                          f"{ind2}{pi} = arith.xori {ph}, %pr_one : i32"]
                    if group == 1:
                        eidx = s
                    else:
                        # the group's empty barrier: completes when the consumer released the group NB groups back
                        # (parity ((c / D) & 1) ^ 1 as for per-stage barriers, since c / (G * NB) == c / D)
                        gq_, eidx = pf("gq"), pf("gb")
                        bl += [f"{ind2}{gq_} = arith.divui {cur}, %pr_G : i32", f"{ind2}{eidx} = arith.remui {gq_}, %pr_NB : i32"]
                    bl += [f"{ind2}{eb} = ttg.memdesc_index {pa['%rg_empty']}[{eidx}] : {ebar_ty} -> {bar1}",
                           f"{ind2}ttng.wait_barrier {eb}, {pi} : {bar1}"]
                    rb_line = f"{ind2}{rb} = ttg.memdesc_index {pa['%rg_ready']}[{s}] : {bar_ty} -> {bar1}"
                    if ring_has_tma:
                        bl.append(rb_line)   # the TMA copies complete on it
                    if ring_has_tma and kind == "w":
                        # one arrival per stage of the global sequence (+ the stage's TMA bytes, 0 if it has none):
                        # Triton's own expect count, the ready barrier counts it once
                        bl.append(f"{ind2}ttng.barrier_expect {rb}, {tma_bytes.get(key, 0)}, %pr_true : {bar1}")
                    for c in cs:
                        ring = view_ring[dst_of(c)]
                        if (ring.name in act_names) != (kind == "a"):
                            continue
                        v = pf("slot")
                        bl += view_line(ind2, v, ring.name, s, ring_ref[ring.name])
                        if c.i in tma_of:
                            m = tma_of[c.i]
                            tys = m.group("tys")
                            parts = re.match(r"^(\S+<[^>]*>),\s*(!ttg\.memdesc<[^>]*>)\s*->\s*(.*)$", tys)
                            if parts is None:
                                raise Fail("TMA copy type list")
                            coords = ", ".join(rn(x.strip()) for x in m.group("coords").split(","))
                            bl.append(f"{ind2}ttng.async_tma_copy_global_to_local {rn(m.group('desc'))}[{coords}] {v}, "
                                      f"{rb}, %pr_true : {parts.group(1)}, {bar1} -> {parts.group(3)}")
                            continue
                        m = copy_of[c.i]
                        cl = P._code(lines[c.i]).strip().replace(f", {m.group('dst')}", ", %RG_SLOT", 1)
                        if m.group("mask"):
                            r_ = resid_of[c.i]
                            cl = re.sub(r"\s+mask\s+" + re.escape(m.group("mask")), f" mask {r_}" if r_ else "", cl)
                            if not r_ and m.group("other"):
                                cl = re.sub(r"\s+other\s+" + re.escape(m.group("other")), "", cl)
                        bl.append(ind2 + rn(cl).replace("%RG_SLOT", v))
                    if not ring_has_tma:
                        bl.append(rb_line)
                    if kind == "a" or not ring_has_tma or w_has_cp:
                        bl.append(f"{ind2}ttng.async_copy_mbarrier_arrive {rb} {{noIncrement}} : {bar1}")
                    nxt = pf("n")
                    if p:
                        inc = pf("i")
                        prod.append(f"{ind}{nxt} = scf.if {rn(p)} -> (i32) {{")
                        prod.extend(bl)
                        prod.extend([f"{ind}  {inc} = arith.addi {cur}, %pr_one : i32", f"{ind}  scf.yield {inc} : i32",
                                     f"{ind}}} else {{", f"{ind}  scf.yield {cur} : i32", f"{ind}}}"])
                    else:
                        prod.extend(bl)
                        prod.append(f"{ind}{nxt} = arith.addi {cur}, %pr_one : i32")
                    cur = nxt
                else:
                    prod.append(ind + rn(code.strip()))
            return cur

        emit_p(body, "%pr_c0", pind)
        prod.append(f"{pind}ttg.warp_return")
        return prod

    # producer warp count: re-layout every distributed tensor type of the partition to pw warps (layouts wrap)
    new_aliases = []
    aliases = dict(re.findall(r"^(#blocked\d*) = (#ttg\.blocked<\{.*\}>)\s*$", text, flags=re.M))

    def relayout(prod, pwn, sfx):
        if pwn == nw:
            return prod
        if nw % pwn:
            raise Fail("producer warps do not divide the warp count")
        ptext = "\n".join(prod)
        if re.search(r"#linear\d*\b|#mma\d*\b|#ttg\.(linear|nvidia_mma)", ptext):
            raise Fail("producer uses a non-blocked distributed layout")
        # (length, name): the emitted alias definitions do not depend on set order (PYTHONHASHSEED)
        used = sorted(set(re.findall(r"#blocked\d*\b", ptext)), key=lambda a: (len(a), a), reverse=True)
        for a in used:
            m = re.search(r"warpsPerCTA = \[([0-9, ]+)\]", aliases[a])
            wv = [int(x) for x in m.group(1).split(",")]
            need_div = nw // pwn
            out_w = []
            for x in wv:
                g = math.gcd(x, need_div)
                out_w.append(x // g)
                need_div //= g
            if need_div != 1:
                raise Fail("cannot scale warps")
            new_aliases.append(f"{a}{sfx} = " + aliases[a].replace(m.group(0), f"warpsPerCTA = [{', '.join(map(str, out_w))}]"))
        return [re.sub(rf"(#blocked\d*)\b(?!{sfx})", rf"\1{sfx}", l) if "#blocked" in l else l for l in prod]
    pwn = pw or nw
    prod = [l.replace("#rg_blk1", "#rg_blk1p") for l in relayout(build("w"), pwn, "_pw")]
    pwa_n = (pwa or pwn) if act_names else 0
    prod_a = relayout(build("a"), pwa_n, "_pa") if act_names else None

    # ---------------- assemble
    pre = [f"{ind0}%rg_c0 = arith.constant 0 : i32"]
    if shared:
        for (name, bt, bmax, rs, align) in byte_rings:
            pre.append(f"{ind0}{name} = ttg.local_alloc {{alignment = {align} : i32}} : () -> {bt}")
            for r in rs:
                pre.append(f"{ind0}{r.name} = ttg.memdesc_reinterpret {name} : {bt} -> {view_ty[r.name]}")
    else:
        for r in sel:
            pre.append(f"{ind0}{r.name} = ttg.local_alloc : () -> {view_ty[r.name]}")
    pre += [f"{ind0}%rg_ready = ttg.local_alloc : () -> {bar_ty}", f"{ind0}%rg_empty = ttg.local_alloc : () -> {ebar_ty}"]
    if tickets:
        pre += [f"{ind0}%rg_mbox = ttg.local_alloc : () -> {mb_ty}", f"{ind0}%rg_mfull = ttg.local_alloc : () -> {mbar_ty}",
                f"{ind0}%rg_mempty = ttg.local_alloc : () -> {mbar_ty}"]
        for s_ in range(MB):
            pre += [f"{ind0}%rg_mcs{s_} = arith.constant {s_} : i32",
                    f"{ind0}%rg_mf{s_} = ttg.memdesc_index %rg_mfull[%rg_mcs{s_}] : {mbar_ty} -> {bar1}",
                    f"{ind0}ttng.init_barrier %rg_mf{s_}, 1 : {bar1}",
                    f"{ind0}%rg_me{s_} = ttg.memdesc_index %rg_mempty[%rg_mcs{s_}] : {mbar_ty} -> {bar1}",
                    f"{ind0}ttng.init_barrier %rg_me{s_}, 1 : {bar1}"]
    post = []
    if wo_nodes:
        pre.append(f"{ind0}%rg_wo = ttg.local_alloc : () -> {wo_ty}")
        for k in range(len(wo_nodes)):
            pre += [f"{ind0}%rg_woc{k} = arith.constant {k} : i32",
                    f"{ind0}%rg_woi{k} = ttg.memdesc_index %rg_wo[%rg_woc{k}] : {wo_ty} -> {bar1}",
                    f"{ind0}ttng.init_barrier %rg_woi{k}, 1 : {bar1}"]
            post.append(f"{ind0}ttng.inval_barrier %rg_woi{k} : {bar1}")
    # ready count: every producer thread's noinc arrival (cp.async completion) + with TMA members, the weight
    # producer's one barrier_expect arrival per stage (its TMA bytes complete the phase), replacing its noinc arrivals
    # when it issues no cp.async at all
    ready_count = 32 * (pwn + pwa_n) if not ring_has_tma else 1 + (32 * pwn if w_has_cp else 0) + 32 * pwa_n
    for s in range(D):
        pre += [f"{ind0}%rg_cs{s} = arith.constant {s} : i32",
                f"{ind0}%rg_r{s} = ttg.memdesc_index %rg_ready[%rg_cs{s}] : {bar_ty} -> {bar1}",
                f"{ind0}ttng.init_barrier %rg_r{s}, {ready_count} : {bar1}"]
        post += [f"{ind0}ttng.inval_barrier %rg_r{s} : {bar1}"]
    for s in range(NB):
        pre += [f"{ind0}%rg_e{s} = ttg.memdesc_index %rg_empty[%rg_cs{s}] : {ebar_ty} -> {bar1}",
                f"{ind0}ttng.init_barrier %rg_e{s}, 1 : {bar1}"]
        post += [f"{ind0}ttng.inval_barrier %rg_e{s} : {bar1}"]
    nparts = 2 if act_names else 1
    regs = f" attributes {{requestedRegisters = array<i32: {', '.join([str(producer_regs)] * nparts)}>}}" if producer_regs else ""
    sig = ", ".join(f"{pa[v]}: {t}" for v, t in zip(cap_all, cap_ty))
    ws = [f"{ind0}ttg.warp_specialize({', '.join(cap_all)}){regs}", f"{ind0}default {{"] + default + \
         [f"{ind0}  ttg.warp_yield", f"{ind0}}}", f"{ind0}partition0({sig}) num_warps({pwn}) {{"] + prod
    if act_names:
        ws += [f"{ind0}}}", f"{ind0}partition1({sig}) num_warps({pwa_n}) {{"] + prod_a
    ws += [f"{ind0}}} : ({', '.join(cap_ty)}) -> ()"]
    prefix = [l for k, l in enumerate(lines[f.start + 1: split_at + 1], f.start + 1) if k not in drop]
    if any(k in edits or k in before or k in after for k in range(f.start + 1, split_at + 1)):
        raise Fail("ring code before the warp_specialize")
    new = lines[: f.start + 1] + prefix + pre + ws + post + [f"{ind0}tt.return", lines[f.end - 1]] + lines[f.end:]
    # callees run in the default partition: WS lowering needs their warp count
    for k, l in enumerate(new):
        if re.match(r"^\s*tt\.func private", l):
            if "attributes {" in l:
                new[k] = l.replace("attributes {", f"attributes {{ws_num_warps = {nw} : i32, ", 1)
            else:
                raise Fail("callee without attribute dict")
    out = "\n".join(new) + "\n"
    out = re.sub(r"^(#smem = .*)$", r"\1\n#rg_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>"
                 r"\n#rg_flat = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>"
                 rf"\n#rg_blk1 = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [{nw}], order = [0]}}>"
                 rf"\n#rg_blk1p = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [{pwn}], order = [0]}}>",
                 out, count=1, flags=re.M)
    if new_aliases:
        out = re.sub(r"^(#rg_bar = .*)$", lambda m_: m_.group(1) + "\n" + "\n".join(new_aliases), out, count=1, flags=re.M)
    if "#rg_bar" not in out.split("module")[0]:
        raise Fail("no #smem alias to anchor the barrier layout")
    if maxnreg:
        out = out.replace('"ttg.num-warps" =', f'"ttg.maxnreg" = {maxnreg} : i32, "ttg.num-warps" =', 1)
    info = dict(rings=[r.name for r in sel], D=D, want_D=max(r.depth for r in sel), tickets=len(tickets), pstages=len(pstages), cstages=len(cinfo), shared=shared,
                act=sorted(act_names), spins=sum(1 for g in spin_groups if any(x.i in spin_ops for x in g)), pcaps=len(pcaps),
                order_points=len(wo_nodes), ws_start=split_at, rolled=len(roll_loops), w_lookahead=w_lookahead, spin_gated=spin_gated,
                ring_bytes=(sum(D * b[2] for b in byte_rings) if shared else sum(D * r.tile_bytes for r in sel)))
    return out, info


def _outside_strings(text, fn):
    """fn applied to the parts of ``text`` outside string literals (inline asm text stays verbatim)."""
    out, at = [], 0
    for m in P._STRING.finditer(text):
        out += [fn(text[at:m.start()]), m.group(0)]
        at = m.end()
    return "".join(out + [fn(text[at:])])


def _unroll_loops(lines, consts, max_trip, marker=None, factor=0):
    """Fully unroll scf.for loops of ``lines`` (a region body) with constant bounds, <= max_trip iterations, no nested
    region in the body and (if given) a line containing ``marker``: iteration k is the body with the induction variable
    replaced by a constant, the iter_args by iteration k-1's yielded values and every value defined in the body renamed
    with suffix _u<k>; uses of the loop results after the loop take the last iteration's yielded values.
    ``factor`` F > 1: a qualifying loop with more than max_trip iterations whose trip count F divides is unrolled by F:
    the loop keeps its bounds with step F * step, its body holds F copies (copy k uses iv + k * step and copy k-1's
    yielded values) and yields the last copy's values, so its results are unchanged."""
    i = 0
    while i < len(lines):
        m = _FOR.match(P._code(lines[i]))
        if not m:
            i += 1
            continue
        ind = len(m.group("ind"))
        j = i + 1
        while j < len(lines) and not (P._code(lines[j]).strip().startswith("}") and P._indent(lines[j]) == ind):
            j += 1
        body = lines[i + 1:j]
        lb, ub, st = (consts.get(m.group(k)) for k in ("lb", "ub", "st"))
        shape_ok = j < len(lines) and body and P._code(body[-1]).strip().startswith("scf.yield") and \
            not any(P._code(x).rstrip().endswith("{") for x in body) and (marker is None or any(marker in x for x in body)) \
            and None not in (lb, ub, st) and st > 0
        trip = len(range(lb, ub, st)) if shape_ok else 0
        full = shape_ok and 0 < trip <= max_trip
        part = shape_ok and not full and factor > 1 and trip > factor and trip % factor == 0
        if not (full or part):
            i += 1
            continue
        ia = re.findall(rf"({V})\s*=\s*({V})", m.group("ia") or "")
        yl = P._code(body[-1]).strip()
        yv = re.findall(V, yl.split(":")[0]) if ia else []
        if len(yv) != len(ia):
            i += 1
            continue
        defs = set()
        for x in body[:-1]:
            defs |= {d.split("#")[0] for d in re.findall(rf"({V})(?::\d+)?\s*=(?!=)", P._code(x))}

        if full:
            cur = dict(ia)
            out = []
            for k, ivv in enumerate(range(lb, ub, st)):
                sub = dict(cur)
                ivc = f"{m.group('iv')}_u{k}"
                sub[m.group("iv")] = ivc
                for d in defs:
                    sub[d] = f"{d}_u{k}"

                def rn(s, sub=sub):
                    return re.sub(V, lambda mm: sub.get(mm.group(0).split("#")[0], mm.group(0).split("#")[0]) +
                                  ("#" + mm.group(0).split("#")[1] if "#" in mm.group(0) else ""), s)
                out.append(" " * ind + f"{ivc} = arith.constant {ivv} : {m.group('ivt')}")
                out += [" " * ind + _outside_strings(x[:len(P._code(x))], rn).strip() for x in body[:-1]]
                cur = {a: rn(v) for (a, _), v in zip(ia, yv)}
            res = m.group("res")
            final = [cur[a] for a, _ in ia]
            rmap = ({f"{res}#{k}": v for k, v in enumerate(final)} if m.group("n") else {res: final[0]}) if res else {}
            tail = lines[j + 1:]
            if rmap:
                tail = [re.sub(V, lambda mm: rmap.get(mm.group(0), mm.group(0)), x) for x in tail]
            lines = lines[:i] + out + tail
            i += len(out)
            continue
        # partial: F copies per trip of a loop with step F * step
        bi = ind + 2
        stc, newst = f"{m.group('iv')}_ufst", f"{m.group('iv')}_ufstep"
        pre = [" " * ind + f"{newst} = arith.constant {st * factor} : {m.group('ivt')}"]
        head = lines[i].replace(f"step {m.group('st')}", f"step {newst}", 1)
        if head == lines[i]:
            i += 1
            continue
        cur = {a: a for a, _ in ia}
        out = []
        for k in range(factor):
            sub = dict(cur)
            if k == 0:
                ivk = m.group("iv")
                ivl = []
            else:
                ivk = f"{m.group('iv')}_uf{k}"
                ivl = [" " * bi + f"{stc}{k} = arith.constant {k * st} : {m.group('ivt')}",
                       " " * bi + f"{ivk} = arith.addi {m.group('iv')}, {stc}{k} : {m.group('ivt')}"]
            sub[m.group("iv")] = ivk
            for d in defs:
                sub[d] = d if k == 0 else f"{d}_uf{k}"

            def rn(s, sub=sub):
                return re.sub(V, lambda mm: sub.get(mm.group(0).split("#")[0], mm.group(0).split("#")[0]) +
                              ("#" + mm.group(0).split("#")[1] if "#" in mm.group(0) else ""), s)
            out += ivl + [" " * P._indent(x) + _outside_strings(x[:len(P._code(x))], rn).strip() for x in body[:-1]]
            cur = {a: rn(v) for (a, _), v in zip(ia, yv)}
        yind = " " * P._indent(body[-1])
        vals, _, tys = yl[len("scf.yield"):].partition(":")
        ylast = f"{yind}scf.yield {', '.join(cur[a] for a, _ in ia)} : {tys.strip()}" if ia else body[-1]
        lines = lines[:i] + pre + [head] + out + [ylast] + lines[j:]
        i += len(pre) + 1 + len(out) + 1
    return lines


_CTX = None


def smem_of(text: str) -> int:
    """Triton's own shared-memory allocator (the leading passes of make_llir) on a TTGIR text."""
    global _CTX
    from triton._C.libtriton import ir, nvidia, passes  # pyrefly: ignore[missing-module-attribute]
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend

    if _CTX is None:
        _CTX = ir.context()
        ir.load_dialects(_CTX)
        CUDABackend(GPUTarget("cuda", 100, 32)).load_dialects(_CTX)
    fd, path = tempfile.mkstemp(suffix=".ttgir")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    try:
        try:
            mod = ir.parse_mlir_module(path, _CTX)
            mod.context = _CTX
            pm = ir.pass_manager(_CTX)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            passes.ttgpuir.add_allocate_warp_groups(pm, False)
            passes.convert.add_scf_to_cf(pm)
            passes.gluon.add_inliner(pm)
            nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, 100, 88)
            pm.run(mod, "ttgir_weight_ring_smem")
        except RuntimeError as e:
            # MLIR rejects the candidate (e.g. a warp-specialized module whose warp count is not a multiple of 4):
            # a refusal of this candidate, so the root-suffix policy goes on with the next suffix (verify5 C3 / C7)
            raise Fail(f"MLIR rejects the candidate ({str(e).splitlines()[0][:120] if str(e) else type(e).__name__})") from e
        check_allocation_alignment(mod.str())
        return int(mod.get_int_attr("ttg.shared"))
    finally:
        os.unlink(path)


def check_allocation_alignment(allocated: str) -> None:
    """After Triton's allocator: every buffer that requested an explicit alignment got an offset that honors it."""
    n = 0
    for line in allocated.splitlines():
        if "ttg.local_alloc" not in line:
            continue
        a = re.search(r"\balignment = (\d+) : i32", line)
        o = re.search(r"\ballocation\.offset = (\d+) : i32", line)
        if a is None:
            continue
        n += 1
        if o is None or int(o.group(1)) % int(a.group(1)):
            raise Fail(f"byte ring placed at offset {o and o.group(1)}, not a multiple of {a.group(1)}")


SMEM_LIMIT = 232448 - 1024

# Shared memory of one B200 (sm_100) SM and the driver's per-CTA reservation: k co-resident CTAs fit while
# k * (smem + SMEM_RESERVED) <= SMEM_PER_SM.
SMEM_PER_SM = 233472
SMEM_RESERVED = 1024


def resident_limit(ctas: int) -> int:
    """Largest per-CTA shared memory that keeps ``ctas`` CTAs per SM co-resident (:data:`SMEM_LIMIT` for <= 1)."""
    return SMEM_LIMIT if ctas <= 1 else min(SMEM_LIMIT, SMEM_PER_SM // ctas - SMEM_RESERVED)


# Register file of one SM (32-bit registers) and the request of the ring's producer partitions when the launch's
# co-resident cohort fixes the register budget (:func:`_resident_registers`): the producers issue copies (TMA / cp.async
# address math, cloned spins and dependent-address cones), the consumer keeps the rest of the budget.
REGS_PER_SM = 65536
PRODUCER_REGS = 40
# Spill-store bytes a resident ring binary may carry: Triton's warp-group dispatch keeps a few thread-index values of
# the function entry live across the producers' / consumer's ``setmaxnreg``, which ptxas spills once at entry (B5-6:
# 12-40 B on the merged MoE MK, stores before the first re-allocation).  Only rings with a register budget
# (:func:`_resident_registers`) get it; every other ring binary that spills falls back as before.
RESIDENT_SPILL_BYTES = 64


def _resident_registers(text: str, ctas: int) -> str:
    """Register budget of a warp-specialized ring kernel with ``ctas`` co-resident CTAs per SM.  The ring launches its
    producer warps on top of the original warps (Triton pads the partitions to whole warp groups), so the per-thread
    budget of the launch, REGS_PER_SM / (ctas * threads), is lower than the original kernel's; the kernel gets it as
    ``ttg.maxnreg`` and the producer partitions request :data:`PRODUCER_REGS` (``setmaxnreg``), which hands the
    remainder of the budget to the consumer warp group."""
    m = re.search(r'"ttg\.num-warps"\s*=\s*(\d+)', text)
    parts = [int(x) for x in re.findall(r"^\s*partition\d+\(.*\)\s+num_warps\((\d+)\)\s*\{\s*$", text, flags=re.M)]
    if m is None or not parts:
        raise Fail("resident ring: warp counts not found")
    if re.search(r'"ttg\.maxnreg"', text.split("tt.func", 1)[0]):
        raise Fail("resident ring: module already has ttg.maxnreg")
    total = int(m.group(1)) + -(-sum(parts) // 4) * 4
    budget = min(256, REGS_PER_SM // (ctas * total * 32)) // 8 * 8
    if budget < PRODUCER_REGS:
        raise Fail(f"resident ring: register budget {budget} below the producer request")
    lines = text.split("\n")
    ws = [k for k, l in enumerate(lines) if re.match(r"\s*ttg\.warp_specialize\(", l)]
    if len(ws) != 1 or "attributes" in lines[ws[0]]:
        raise Fail("resident ring: warp_specialize op not found / already annotated")
    lines[ws[0]] += f" attributes {{requestedRegisters = array<i32: {', '.join([str(PRODUCER_REGS)] * len(parts))}>}}"
    out = "\n".join(lines)
    return out.replace('"ttg.num-warps" =', f'"ttg.maxnreg" = {budget} : i32, "ttg.num-warps" =', 1)


# Occupancy: a ring keeps the module's CTAs per SM, all of them (R6-7). The ring binary is judged after ptxas on its
# real resources against its fallback's (ttgir_weight_prefetch._Fallback: registers x threads, shared memory, threads,
# TMEM); this shared-memory rule is the same rule on what the TTGIR tells (no registers yet): the ring's buffers may not
# lower the module's shared-memory occupancy. Measured (build5 R6-7, B200, one tile per CTA, 4 sessions on GPUs 0-3):
# rings at 6 of 9 CTAs/SM lose (+0.8 % at K 2048, +4.3 % at K 4096 for a BN16 GEMV) while others at 5 of 6 win, so
# "2 CTAs/SM are enough" (the former cap of this rule at 2) does not hold; B5-8 (2 -> 1 CTA/SM: +14.9 us) is the same
# rule's extreme.


# CTAs per SM a spin-gated ring keeps at least (when its module had more): the consumer waits on a readiness spin
# before the ring's first stage while the weight producer streams, so the module's CTAs above this would mostly spin
# too (build5 R6-8, B200: Helion GEMV megakernels at 6 of 9 and 4 of 9 / 6 CTAs per SM: -2 % .. -11 % vs the hoist).
SPIN_GATED_KEPT = 2


def kept_occupancy(text: str) -> int:
    """CTAs per SM a ring must keep on a module: its shared-memory occupancy as is (``SMEM_PER_SM // (smem +
    SMEM_RESERVED)``), capped by what a warp-specialized kernel of it can reach at all (the producers add at least one
    warp group: 2048 threads / (32 * (num_warps + 4))); beyond that cap no ring keeps the module's occupancy and the
    post-ptxas comparison with the fallback refuses it."""
    m = re.search(r'"ttg\.num-warps"\s*=\s*(\d+)', text)
    nw = int(m.group(1)) if m else 4
    occ = SMEM_PER_SM // (smem_of(text) + SMEM_RESERVED)
    return max(1, min(occ, 2048 // (32 * (nw + 4))))


def _with_minctasm(text: str, ctas: int) -> str:
    """The entry function asks ptxas for ``ctas`` CTAs per SM (``nvvm.minctasm`` -> ``.minnctapersm``): the warp-
    specialized kernel launches more warps than the original, and the launch's co-resident cohort must still fit the
    register file (ptxas then caps the registers per thread; a binary that would spill falls back)."""
    lines = text.split("\n")
    for k, line in enumerate(lines):
        if re.match(r"\s*tt\.func\s+public\s+@", line):
            if "nvvm.minctasm" in line:
                raise Fail("entry function already carries nvvm.minctasm")
            new, n = re.subn(r"attributes \{", f"attributes {{nvvm.minctasm = {ctas} : i32, ", line, count=1)
            if n != 1:
                raise Fail("entry function without an attribute dictionary")
            lines[k] = new
            return "\n".join(lines)
    raise Fail("no public entry function")


# Code-offset knee of B200 (sm_100) instruction fetch (bytes of SASS): executed code that lies beyond the first ~64 KB of
# a kernel's binary is fetched slowly when it is not resident, ~0.2 us per KB of executed code past the knee (ATT6 PAD:
# MK16 with never-executed padding before its code: 63.2 KB +0.0, 66.4 KB +0.3, 69.5 KB +1.2, 78.9 KB +3.0 us; the same
# padding after its code +0.2; ATT6 verify: the same knee on another GPU and in ncu PC sampling, 44-87 % no-instruction
# stalls past 63 KB vs 0-2 % below).  A ring form's twins differ only in the ring's own code (producer partitions and the
# consumer root, the kernel's suffix), so once the binary exceeds the knee every byte a bigger twin adds is code past the
# knee, and its unrolled consumer runs each copy once per trip instead of re-running one resident loop body.  The stages
# hook therefore walks a form's smaller twins (rolled producers, then rolled producers without the consumer unroll, see
# :data:`SIZE_LADDER`) while the binary exceeds the knee; below it the unrolled form stays (V1d ring, 40 KB: rolling costs
# +0.1..+0.2 us).  This replaces the former total-size budget of 103 KB (RING3 / ATT5 probes: idle 103 KB +0, pw2 111 KB
# +0.64 us, u0 118 KB +1.2 us), a proxy for the same knee that let a 95-102 KB attention MK keep the unrolled consumer
# twin (ATT7 S2: ring_roll 101.5 KB vs ring_roll_u0 95.0 KB of the same kernel: u0 -0.05 / -0.25 us std fi / plain).
CODE_BUDGET = 64 * 1024

# The ring options of the default path.  Every option is an automatic decision (no environment variable, no per-kernel
# input); the measurements behind each (B200, paired, byte-exact arms; NOTES of the ring, ring2, ring3, att5 tracks):
# * act=True - activation rings move into a second producer partition that clones the readiness spins guarding them,
#   so the consumer keeps no cp.async pipeline: its K loop has 1 mbarrier try-wait + 1 release per stage instead of
#   ~2 bar.sync + ~3 cp.async waits (RING2 S4).  V1d q-prep MK: two-producer ring -0.22..-0.65 us vs the hoist in
#   12/12 seed x mode cells, weight-only ring -0.05 on average; attention v4g O root: weight-only +2.80 / +3.82 vs
#   act +2.50 / +2.98 (RING3 d2).  An activation rule that refuses falls back to the weight-only ring (here and as the
#   ``ring_wonly`` candidate of the stages hook).
# * pw="auto", pwa="auto" - one producer warp each when the ring has an activation producer, else the kernel's warp
#   count for the weight producer.  V1d: 1 + 1 warps beat 4 + 4 in every cell (4 + 4 loses +0.40 / +0.45 on two seeds,
#   RING2 s4v); chain3 best 9.93 vs 11.15 us; weight-only streams with many short tasks want the issue width of the
#   full warp count (pw 4 beats pw 1 by 0.1-1.0 us, RING2 s7s).
# * unroll=32 - the pass fully unrolls consumer loops of at most 32 iterations with a ring stage (ptxas does not unroll
#   a loop holding the mbarrier try-wait loop): chain3 best-vs-best -0.49 us with it (RING2 s7c), BN16 BK64 -1.21 us
#   with 32 vs -0.52 with 16 (s9c).  On a kernel past CODE_BUDGET (the 64 KB code-offset knee) the unrolled code costs
#   more than it saves (ATT5: attention MK full unroll + roll +2.08 vs no unroll + roll +1.23 us), so the last size twin
#   drops it.
# * late_ws=True - the warp_specialize starts at the first top-level op holding a ring stage, so ring-free leading roots
#   run before the partitions exist.  V1d weight-only ring: +0.47 / +0.35 us vs the hoist with the warp_specialize at
#   the entry, -0.32 / -0.27 with it late (RING2 s2v); chain3: late better in 5 of 6 configs (s2c).  The two-producer
#   ring and every root-suffix ring imply it.
# * depth - the deepest original pipeline the ring moves (Triton's num_stages for the loop), lowered until the ring fits
#   shared memory: V1d D4 +0.4..+1.5 us and D8 +0.1..+1.2 us vs D6 (RING2 s3v / s4v); chain3 D8 -0.18 at one config
#   but +1.8 at another (s6c).
# * group=1 - one empty barrier and release per stage: release groups (g2, d8g4, d4g2) lose on V1d (+0.26..+1.1 us vs
#   g1) and on chain3 (RING2 s2v / s2c).
# * prefetch=False - the register prefetch of a single-stage loop ties on chain3 (within 0.03 us) and loses on V1d in
#   act mode (+0.27..+0.44 us, RING2 s3c / s4v).
# * drop_waits=False - a weight-only consumer keeps its own activation pipeline, whose waits order it; in act mode the
#   consumer issues no cp.async and its vestigial waits are dropped regardless (RING2 S4).
# * unroll_factor=0 - partial unrolling ties with no unroll (uf2 +1.64 / +2.26 vs u0 +1.65 / +2.50) or loses (uf4) on
#   the attention MK (RING3 d2), where the size ladder decides.
# Removed: lane-0 polling in the activation producer (refuted, RING2 H3 / RING3 d4: +0.04..+0.1 us; its assembly is not
# on the hoist pass's allowlist).
RING_DEFAULTS = {"act": True, "pw": "auto", "pwa": "auto", "unroll": 32, "late_ws": True}

# Size twins of a ring form, in the order the stages hook walks them while the binary's code exceeds CODE_BUDGET:
# (label suffix, options): rolled producers (ATT5 / ATT6: the producer issues the pipeliner's prologue tiles from its
# loop clone; 115 -> 106 KB on the attention MK), then rolled producers without the consumer unroll (attention MK
# 125.6 -> 105.9 KB, the best ring form there).  A twin is a candidate only where it changes the ring.
SIZE_LADDER = (("", {}), ("_roll", {"roll": True}), ("_roll_u0", {"roll": True, "unroll": 0}))

# Profitability: a ring form is a candidate only if its producer partitions issue at least this share of the module's
# global->shared async copy bytes (cp.async and TMA copies, tile bytes x constant loop trips; :func:`copy_coverage`).
# The ring turns the whole persistent kernel into a warp-specialized one (8-warp CTAs whose producer warps idle outside
# the ring's roots; a null warp_specialize alone costs +0.07..+0.4 us, RING2 s7a / RING3 d3); it pays for that only
# through the copies it streams.  Measured (trip-weighted shares): the attention megakernel's O-root ring (50%) wins
# -0.1..-0.3 us warm and -1.2 us code-cold (ATT5 R-MAIN), the V1d q-prep ring (100%) -0.44..-0.51 us (INTEG C2); the
# merged GLM MoE megakernel's shared-down (R3F) ring loses: its cp.async-only form (2% in the old per-op count)
# +0.4..+0.5 us/layer (integ verify job vr), its TMA form (11-12%) +0.3 us dynamic / +1.3 us static (build5 B5-8).
MIN_COPY_COVERAGE = 0.25

_COPY_DST = re.compile(r"->\s*(?:!ttg\.memdesc)?<((?:\d+x)+)([A-Za-z0-9]+),")


_COV_FOR = re.compile(rf"scf\.for\s+{V}\s*=\s*({V})\s+to\s+({V})\s+step\s+({V})")
_COV_CONST = re.compile(rf"^\s*({V})\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i(?:32|64)\b")


def copy_coverage(text: str) -> float:
    """Share of the global->shared async copy bytes of a module that the producer partitions of its warp_specialize
    issue; 0.0 without copies.  Each copy op counts its tile bytes times the trip counts of its enclosing scf.for loops
    with constant bounds (a pipelined K loop's in-loop copy issues one tile per iteration; loops with bounds known only
    at run time - task loops - count once)."""
    consts = {}
    for line in text.splitlines():
        m = _COV_CONST.match(line)
        if m:
            consts.setdefault(m.group(1), int(m.group(2)))
    total = moved = 0
    in_part = False
    ws_ind = None
    loops = []   # (indentation of the scf.for line, trip count)
    for line in text.splitlines():
        s = line.strip()
        ind = len(line) - len(line.lstrip(" "))
        if loops and s.startswith("}") and ind == loops[-1][0]:
            loops.pop()
        m = _COV_FOR.search(P._code(line))
        if m:
            lb, ub, st = (consts.get(m.group(k)) for k in (1, 2, 3))
            trip = -(-(ub - lb) // st) if None not in (lb, ub, st) and st > 0 and ub > lb else 1
            loops.append((ind, max(1, trip)))
        if "ttg.warp_specialize(" in s:
            ws_ind = ind
        elif ws_ind is not None and ind == ws_ind:
            if re.match(r"partition\d+\(", s):
                in_part = True
            elif s.startswith("default {") or s.startswith("} :"):
                in_part = False
        if P._COPY not in s and P._TMA_COPY not in s:
            continue
        m = _COPY_DST.search(P._code(line))
        if m is None:
            continue
        b = math.prod(int(d) for d in m.group(1).rstrip("x").split("x")) * _BITS.get(m.group(2), 8) // 8
        b *= math.prod(t for _, t in loops)
        total += b
        moved += b if in_part else 0
    return moved / total if total else 0.0


# TEST ONLY (not a setting, not documented for users): a negative control that must produce wrong results
# (``neg`` of :func:`rewrite`), read when a kernel compiles and part of the compile key while set.
_NEG_ENV = "HELION_TTGIR_RING_NEG"


def ring_options() -> dict:
    """The options of the default path (:data:`RING_DEFAULTS`), plus the private negative-control test hook."""
    out = dict(RING_DEFAULTS)
    neg = os.environ.get(_NEG_ENV)
    if neg:
        out["neg"] = neg
    return out


# Old name (tools of other tracks import it); it reads no knob but the private test hook.
options_from_env = ring_options


def decisions_key() -> str:
    """Digest of the automatic decisions' current values (:data:`CODE_BUDGET`, :data:`SIZE_LADDER`,
    :data:`MIN_COPY_COVERAGE`, :data:`RING_DEFAULTS`, :data:`SMEM_LIMIT`).  The stages hook puts it in the compile key of
    every ring compile: the binary depends on these values, and the module digest covers only the source, not a value a
    tool or test patches at run time (ATT6 verify, mutant K15)."""
    blob = repr((CODE_BUDGET, SIZE_LADDER, MIN_COPY_COVERAGE, sorted(RING_DEFAULTS.items()), SMEM_LIMIT, SMEM_PER_SM,
                 SMEM_RESERVED, REGS_PER_SM, PRODUCER_REGS, RESIDENT_SPILL_BYTES, "occupancy=module",
                 SPIN_GATED_KEPT))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def options_key() -> str:
    """The ring's part of the compile key beyond the module digest: empty unless the test hook is set."""
    neg = os.environ.get(_NEG_ENV)
    return f"neg={neg}" if neg else ""


def rewrite_budgeted(text: str, limit: int | None = None, shared: bool = True, **opts) -> tuple[str, dict]:
    """The ring for a suffix of the kernel's roots, at the deepest depth that fits ``limit`` bytes of shared memory.

    Root-suffix policy (automatic, no per-kernel input): the candidate suffixes start at each top-level op that holds a
    ring copy, longest first.  The first suffix whose ring fits at its FULL depth (the deepest original ring it moves,
    i.e. never shallower than Triton's own pipelines) wins; the rings of the roots before it keep Triton's pipelines and
    get the canonical hoist (:func:`ttgir_weight_prefetch.rewrite` runs first on the whole module; the ring then takes
    the suffix's copies wherever the hoist put them).  The ring's buffers are allocated where the warp_specialize starts,
    so Triton's allocator overlays them on the prefix roots' dead buffers.  If no suffix fits at full depth, the longest
    suffix that fits at all (lowered depth).  For each suffix the two-producer ring (``act``) is tried first, then the
    weight-only ring when an activation rule refuses.  A suffix is skipped when its ring streams less than
    :data:`MIN_COPY_COVERAGE` of the module's copy bytes, or when its first root gives the weight producer no lookahead
    (rule E clones every readiness spin the consumer passes before that root's first stage, and W cannot run on into
    later tasks of a loop past spins that poll only state published outside the kernel).

    Occupancy: the ring never lowers the module's shared-memory occupancy (:func:`kept_occupancy`): the limit is at
    most :func:`resident_limit` of it.  ``resident_ctas`` k > 1 (the launch's co-resident cohort per SM): the ring keeps that
    occupancy - its shared memory
    fits :func:`resident_limit` (k) (the depth is lowered until it does, as for the device limit) and the entry function
    carries ``nvvm.minctasm = k``; its register budget is the cohort's (:func:`_resident_registers`)."""
    opts = {**ring_options(), **opts}
    ctas = int(opts.pop("resident_ctas", 0) or 0)
    if not opts.pop("_occupancy_kept", False):
        # occupancy rule: the ring keeps the module's CTAs per SM (DESIGN 9.2; B5-8: the merged MoE MK's dynamic grid
        # with a D4 ring at 1 CTA/SM instead of 2 lost +14.9 us; R6-7: 6 instead of 9 loses on a GEMV), except a
        # spin-gated ring, which keeps min(module, SPIN_GATED_KEPT) (R6-8; the same rule as the post-ptxas guard)
        occ = kept_occupancy(text)
        kept = dict(opts, _occupancy_kept=True, resident_ctas=ctas)
        if occ > SPIN_GATED_KEPT:
            try:
                t, info = rewrite_budgeted(text, min(limit or SMEM_LIMIT, resident_limit(SPIN_GATED_KEPT)), shared,
                                           **kept)
                if info.get("spin_gated"):
                    return t, info
            except Fail:
                pass
        if occ > 1:
            limit = min(limit or SMEM_LIMIT, resident_limit(occ))
        return rewrite_budgeted(text, limit, shared, **kept)
    if ctas > 1:
        limit = min(limit or SMEM_LIMIT, resident_limit(ctas))
        t, info = rewrite_budgeted(text, limit, shared, **opts)
        return _with_minctasm(_resident_registers(t, ctas), ctas), dict(info, resident_ctas=ctas)
    opts.pop("_occupancy_kept", None)
    if "from_top" in opts:
        return _act_or_wonly(text, limit, shared, opts)
    starts = _ring_starts(text)
    if not starts:
        raise Fail("no movable ring")
    hoisted = None
    best, errors = None, []
    for k, s in enumerate(starts):
        src = text
        if k:
            if hoisted is None:
                try:
                    h, rep = P.rewrite(text, opts.get("alias_classes", ()))
                    hoisted = h if rep.hoisted else text
                except Exception:  # noqa: BLE001 - no hoist: the prefix roots keep their plain pipelines
                    hoisted = text
            src = hoisted
            s = _ring_starts(src)[k] if len(_ring_starts(src)) == len(starts) else s
        try:
            t, info = _act_or_wonly(src, limit, shared, dict(opts, from_top=s if k else None))
        except Fail as e:
            errors.append(f"from {s}: {e}")
            continue
        if not info.get("w_lookahead", True):
            errors.append(f"from {s}: the weight producer clones every spin before the first stage and no loop lets it "
                          "run past them into later tasks (no lookahead)")
            continue
        cov = copy_coverage(t)
        if cov < MIN_COPY_COVERAGE:
            errors.append(f"from {s}: moves {cov:.3f} of the copy bytes (< MIN_COPY_COVERAGE)")
            continue
        info = dict(info, from_top=(s if k else None), prefix_hoisted=bool(k and hoisted is not text))
        if info["D"] >= info["want_D"]:
            return t, info
        if best is None:
            best = (t, info)
    if best is not None:
        return best
    raise Fail("; ".join(errors) or "no ring")


_GENERIC_OP = re.compile(r'^\s*(?:%[\w.$-]+(?::\d+)?(?:\s*,\s*%[\w.$-]+(?::\d+)?)*\s*=\s*)?"([\w.]+)"\(')


def _generic_op(line):
    """Name of an op printed in MLIR's generic form (``%r = "tt.reduce"(%x) ...``), else None."""
    m = _GENERIC_OP.match(line)
    return m.group(1) if m else None


def _ring_starts(text):
    """Indices of the entry function's top-level ops that hold a global->shared copy (cp.async or TMA)."""
    lines = P._uniquify(text).splitlines()
    funcs = P._functions(lines)
    entry = [g for g in funcs if re.search(r"tt\.func\s+public", lines[g.start])]
    if len(entry) != 1:
        return []
    f = entry[0]
    body = _parse_block(lines, f.start + 1, f.end - 1, None)
    return [k for k, n in enumerate(body) if any(x.op in (P._COPY, P._TMA_COPY) for x in _walk([n]))]


def _act_or_wonly(text, limit, shared, opts):
    try:
        return _budgeted(text, limit, shared, dict(opts))
    except Fail as e:
        if not opts.get("act") or opts.get("neg"):
            raise
        # an activation rule refused (or the 2-producer ring does not fit): the weight-only ring may still apply
        out = _budgeted(text, limit, shared, {**opts, "act": False})
        return out[0], dict(out[1], act_refused=str(e))


def _budgeted(text, limit, shared, opts):
    limit = limit or SMEM_LIMIT
    fixed = opts.pop("depth", None)
    g = opts.get("group", 1)
    t, info = rewrite(text, shared=shared, depth=fixed, **opts) if fixed else rewrite(text, shared=shared, **_with_group(opts, None, text))
    d = info["D"]
    while True:
        s = smem_of(t)
        if s <= limit:
            return t, dict(info, smem=s, opts=opts)
        if fixed:
            raise Fail(f"ring does not fit ({s} > {limit})")
        d -= g
        if d < max(2, g):
            raise Fail(f"ring does not fit ({s} > {limit})")
        t, info = rewrite(text, shared=shared, depth=d, **opts)


def _with_group(opts, depth, text):
    """Without a fixed depth, round the default depth down to a multiple of the group (at least one group)."""
    g = opts.get("group", 1)
    if g == 1:
        return opts
    t, info = rewrite(text, shared=True, **{**opts, "group": 1})
    d = max(g, info["D"] - info["D"] % g)
    return {**opts, "depth": d}
