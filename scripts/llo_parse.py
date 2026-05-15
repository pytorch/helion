"""Minimal LLO dump parser for the TPU roofline predictor.

Forked from msl-tpu-kernel's `tools/llo_trace_viewer/llo_tool.py` (Google/Meta),
keeping only the bits we need (bundle parsing, trip-count inference, lane
utilization). We strip the HTML viewer, JSON export, and visualization
pipeline. We also fix a bug where `extract_trip_counts` can return negative
trip counts for kernels with dynamic-bound loops (e.g. ragged_paged_attention),
clamping to 1 and exposing a confidence flag.

Returns:
  parse_llo_dump(dump_dir) → LloParseResult

The result is intended to be the canonical input to `tpu_roofline.simulate_*`.
Eliminates the `--dynamic-bundles` flag and the separate `llo_tool collect`
step.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import re


@dataclass
class LloParseResult:
    """Result of parsing one LLO kernel dump.

    Fields:
      bundles_file, util_file: source paths
      static_bundles: cycles in one kernel body execution
      inferred_trip_count: loop iter count derived from sphi/scmp annotations
      pipeline_depth: software pipelining depth
      dynamic_bundles: static_bundles × inferred_trip_count
      loop_inference_confident: True if sphi `iter bound = N` annotations
        anchored the trip count; False if we fell back to scmp.ge heuristic
        (less reliable, especially for dynamic-bound fori_loops).
      capacity: per-lane slot capacity (e.g. MXU=2, VALU=4)
      lane_names: aligned with capacity
      util_rows: per-bundle slot counts; len == static_bundles
      vmatmul_count, vpop_count: dynamic counts (static × trip)
      lb_nesting_depth: max `LB:` depth seen
    """

    bundles_file: Path
    util_file: Path
    static_bundles: int
    inferred_trip_count: int
    pipeline_depth: int
    dynamic_bundles: int
    loop_inference_confident: bool
    capacity: list[int] = field(default_factory=list)
    lane_names: list[str] = field(default_factory=list)
    util_rows: list[list[int]] = field(default_factory=list)
    vmatmul_count: int = 0
    vpop_count: int = 0
    lb_nesting_depth: int = 0


_BUNDLE_LINE_RE = re.compile(r"\s*(0x[0-9a-fA-F]+|\d+)\s+(.*?):\s*(>*)\s*\{(.*)\}")
_COMMENT_INLINE_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_PHI_BOUND_RE = re.compile(r"iter bound = (\d+)")
_PHI_REG_RE = re.compile(r"(%s\S+)\s*=\s*sphi")
_SADD_RE = re.compile(r"(%s\S+)\s*=\s*sadd\.s32\s+1,\s+(%s\S+)")
_SMOV_RE = re.compile(r"(%s\S+)\s*=\s*smov\s+\([^,]+,\s+(%s\S+)\)")
_SCMP_RE = re.compile(r"scmp\.ge\.s32\.\w+\s+(%s\S+),\s+(\d+)")


def _find_kernel_files(dump_dir: Path) -> tuple[Path, Path]:
    """Locate the bundles + utilization files for the largest kernel in dump_dir.

    Accepts two layouts:
    A. Curated `llo/<entry>/final_bundles.txt` + `utilization.txt`
    B. Raw libtpu dump dir with many timestamped files
    """
    simple_b = dump_dir / "final_bundles.txt"
    simple_u = dump_dir / "utilization.txt"
    if simple_b.is_file() and simple_u.is_file():
        return simple_b, simple_u

    candidates = [
        p
        for p in dump_dir.iterdir()
        if p.name.endswith("-final_bundles.txt") and "schedule-analysis" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No *-final_bundles.txt in {dump_dir}")
    bundles = max(candidates, key=lambda p: p.stat().st_size)

    m = re.match(r"(.+)-(\d+)-final_bundles\.txt$", bundles.name)
    if m:
        prefix = m.group(1)

        def _pass_idx(p: Path) -> int:
            mm = re.search(r"-(\d+)-final_hlo", p.name)
            return int(mm.group(1)) if mm else 0

        utils = sorted(
            dump_dir.glob(f"{prefix}-*-final_hlo-static-per-bundle-utilization.txt"),
            key=_pass_idx,
            reverse=True,
        )
        if utils:
            return bundles, utils[0]
    raise FileNotFoundError(f"No matching utilization file for {bundles.name}")


def _parse_utilization(util_file: Path) -> tuple[list[str], list[int], list[list[int]]]:
    """Parse a *-final_hlo-static-per-bundle-utilization.txt file."""
    text = util_file.read_text().splitlines()
    # Tolerate the compiler's "CAPACTIY" typo.
    if not text or ("CAPACITY" not in text[0] and "CAPACTIY" not in text[0]):
        raise ValueError(f"{util_file}: missing CAPACITY/CAPACTIY header")
    lane_names = [s.strip() for s in text[1].split(",")]
    capacities = [int(s) for s in text[2].split()]
    if "UTILIZATION" not in text[3]:
        raise ValueError(f"{util_file}: missing UTILIZATION marker")
    rows: list[list[int]] = []
    for line in text[4:]:
        vals = line.strip().split()
        if len(vals) == len(lane_names):
            rows.append([int(v) for v in vals])
    return lane_names, capacities, rows


def _parse_bundles(
    bundles_file: Path,
) -> tuple[
    dict[int, list[str]],  # raw instruction strings per bundle index
    dict[int, dict],  # structure: {depth, flags}
    int,  # max LB-nesting depth
]:
    """Parse final_bundles.txt; returns raw instructions + per-bundle structure."""
    raw_instrs: dict[int, list[str]] = {}
    structure: dict[int, dict] = {}
    max_lb_depth = 0
    with bundles_file.open() as f:
        for line in f:
            line = line.rstrip()
            m = _BUNDLE_LINE_RE.match(line)
            if not m:
                continue
            addr_str = m.group(1)
            addr = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str)
            flags = m.group(2).strip()
            depth = len(m.group(3))
            body = m.group(4)
            structure[addr] = {"depth": depth, "flags": flags}
            if "LB" in flags:
                max_lb_depth = max(max_lb_depth, depth)
            raw_texts = [t.strip() for t in body.split(";;") if t.strip()]
            raw_instrs[addr] = raw_texts
    return raw_instrs, structure, max_lb_depth


def _extract_trip_count(
    raw_instrs: dict[int, list[str]], structure: dict[int, dict]
) -> tuple[int, int, bool]:
    """Infer loop iter count from sphi/sadd/scmp annotations.

    Returns (trip_count, pipeline_depth, confident).

    Strategy (same as msl-tpu-kernel's llo_tool):
      1. Find an `LB:` bundle and read `iter bound = N` comments off its sphi
         instructions. Each sphi binds a register to a trip count for ONE
         loop level.
      2. Trace each sphi register through the schedule:
            sphi → sadd.s32 +1 → smov (optional) → scmp.ge.s32 CONST
         where CONST is the actual loop count + pipeline depth.
      3. Match scmp.ge constants back to their phi's iter bound to pin
         down per-level trip counts. Product = linearized total.
      4. Fallback: max(scmp.ge consts) − 2 (assumes pipeline depth 2).
         **Fix**: clamp result to ≥ 1 — the original tool can return
         negative trip counts for kernels with dynamic-bound loops
         (no static scmp.ge consts) leading to `Loop bounds: [-2]`.

    Confidence:
      True  — Step 1/2/3 succeeded (sphi annotations matched).
      False — Fell back to Step 4 (heuristic, often unreliable for
              ragged_paged_attention-style data-dependent loops).
    """
    if not structure:
        return 1, 0, False
    all_addrs = sorted(structure.keys())

    # Find a loop body bundle (LB flag preferred, otherwise first depth > 0)
    loop_start = None
    for addr in all_addrs:
        if "LB" in structure[addr]["flags"]:
            loop_start = addr
            break
    if loop_start is None:
        for addr in all_addrs:
            if structure[addr]["depth"] > 0:
                loop_start = addr
                break
    if loop_start is None:
        return 1, 0, False

    # Step 1: harvest `iter bound = N` from sphi instructions
    phi_to_bound: dict[str, int] = {}
    for raw in raw_instrs.get(loop_start, []):
        bm = _PHI_BOUND_RE.search(raw)
        rm = _PHI_REG_RE.search(raw)
        if bm and rm:
            phi_to_bound[rm.group(1).lstrip("%")] = int(bm.group(1))

    # Step 2: trace sphi → sadd +1
    sadd_to_bound: dict[str, int] = {}
    for addr in all_addrs:
        for raw in raw_instrs.get(addr, []):
            clean = _COMMENT_INLINE_RE.sub("", raw).strip()
            m = _SADD_RE.search(clean)
            if m:
                src = m.group(2).lstrip("%")
                if src in phi_to_bound:
                    sadd_to_bound[m.group(1).lstrip("%")] = phi_to_bound[src]

    # Step 3: trace sadd → smov
    smov_to_bound: dict[str, int] = {}
    for addr in all_addrs:
        for raw in raw_instrs.get(addr, []):
            clean = _COMMENT_INLINE_RE.sub("", raw).strip()
            m = _SMOV_RE.search(clean)
            if m:
                src = m.group(2).lstrip("%")
                if src in sadd_to_bound:
                    smov_to_bound[m.group(1).lstrip("%")] = sadd_to_bound[src]
    all_tracked = {**sadd_to_bound, **smov_to_bound}

    # Step 4: match scmp.ge constants to bounds
    level_to_trip: dict[int, int] = {}
    all_scmp_ge: list[int] = []
    for addr in all_addrs:
        for raw in raw_instrs.get(addr, []):
            clean = _COMMENT_INLINE_RE.sub("", raw).strip()
            m = _SCMP_RE.search(clean)
            if m:
                reg = m.group(1).lstrip("%")
                const = int(m.group(2))
                all_scmp_ge.append(const)
                if reg in all_tracked:
                    level = all_tracked[reg]
                    if level not in level_to_trip or const > level_to_trip[level]:
                        level_to_trip[level] = const

    if level_to_trip:
        linearized = 1
        for trip in level_to_trip.values():
            linearized *= trip
        max_scmp = max(all_scmp_ge) if all_scmp_ge else linearized
        pipeline_depth = max(0, max_scmp - linearized)
        return max(linearized, 1), pipeline_depth, True

    # Fallback: heuristic from max scmp.ge constant. The original tool
    # uses `max_scmp - 2` here, but for kernels with multi-level loops
    # or dynamic bounds this heuristic is unreliable (we've seen it
    # both UNDER-shoot — RPA's `[-2]` — and OVER-shoot — bcast 1×1 by
    # 3× and 1×128 by ~3×). We surface the heuristic value in `confident=False`
    # mode but default to trip=1 (just the static body), forcing the
    # caller to either trust the parse or supply --dynamic-bundles.
    if all_scmp_ge:
        max_val = max(all_scmp_ge)
        pipeline_depth = 2
        linearized_heuristic = max(max_val - pipeline_depth, 1)
        return linearized_heuristic, pipeline_depth, False

    return 1, 0, False


def _count_mxu_vpop(raw_instrs: dict[int, list[str]]) -> tuple[int, int]:
    """Count vmatmul and vpop instructions in the static body (one iter)."""
    vmatmul = 0
    vpop = 0
    for texts in raw_instrs.values():
        for text in texts:
            tl = text.lower()
            if "vmatmul" in tl:
                vmatmul += 1
            elif "vpop" in tl and "eup" not in tl:
                vpop += 1
    return vmatmul, vpop


def parse_llo_dump(dump_dir: Path) -> LloParseResult:
    """Parse one LLO dump directory and return a structured result.

    Handles both curated `llo/<entry>/` dirs (with `final_bundles.txt` +
    `utilization.txt`) and raw libtpu dump dirs.

    dynamic_bundles is computed correctly per llo_tool's model:
      - bundles at depth 0 (prologue/epilogue) execute ONCE
      - bundles at depth > 0 (loop body) execute (trip + pipeline_depth)
    Total dynamic = sum of per-bundle exec counts. Simply multiplying
    static_bundles × trip overshoots for kernels with sizable
    prologue/epilogue (e.g. bcast variants where the inner fori_loop
    is just a fraction of the schedule).
    """
    dump_dir = Path(dump_dir)
    bundles_file, util_file = _find_kernel_files(dump_dir)
    lane_names, capacity, util_rows = _parse_utilization(util_file)
    raw_instrs, structure, lb_depth = _parse_bundles(bundles_file)
    trip, pipe, confident = _extract_trip_count(raw_instrs, structure)
    vmatmul_static, vpop_static = _count_mxu_vpop(raw_instrs)
    static_bundles = len(util_rows)

    # Per-bundle exec counts: prologue/epilogue (depth=0) runs once;
    # loop-body (depth>0) runs trip + pipeline_depth times.
    inner_exec = trip + pipe
    sorted_addrs = sorted(structure.keys())
    dynamic_bundles = 0
    loop_body_bundles = 0
    for addr in sorted_addrs:
        depth = structure[addr]["depth"]
        if depth > 0:
            dynamic_bundles += inner_exec
            loop_body_bundles += 1
        else:
            dynamic_bundles += 1
    # vmatmul/vpop counts: assume they occur in loop body only (most do).
    return LloParseResult(
        bundles_file=bundles_file,
        util_file=util_file,
        static_bundles=static_bundles,
        inferred_trip_count=trip,
        pipeline_depth=pipe,
        dynamic_bundles=dynamic_bundles,
        loop_inference_confident=confident,
        capacity=capacity,
        lane_names=lane_names,
        util_rows=util_rows,
        vmatmul_count=vmatmul_static * inner_exec,
        vpop_count=vpop_static * inner_exec,
        lb_nesting_depth=lb_depth,
    )


__all__ = ["LloParseResult", "parse_llo_dump"]
