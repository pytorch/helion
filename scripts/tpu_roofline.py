"""Roofline runtime predictor for TPU v7x kernels using LLO dumps.

Predicts kernel execution time from a static LLO dump (`*-final_bundles.txt`
and `*-final_hlo-static-per-bundle-utilization.txt`) plus a user-supplied
HBM byte count.

Constants are calibrated against TPU v7x (Ironwood), 1 TensorCore:
  - Clock:           2.2 GHz (msl-tpu-kernel canonical)
  - BF16 peak:       1,153 TFLOPS per TensorCore
  - HBM peak:        3,690 GB/s per TensorCore
  - HBM effective:   3,160 GB/s (~85.6% of peak, measured on jnp.add)
  - Per-call floor:  ~6 µs additive overhead at shapes large enough to
                     exceed memory_floor; can be 20-30 µs at tiny shapes.

The model:
  predicted_us = OVERHEAD_US + max(
      compute_floor_us,           # = dynamic_bundles / clock_ghz / 1000
      memory_floor_us,            # = bytes / hbm_effective_gbps × 1000
  )

Slot analysis decomposes the schedule into per-lane busy cycles so we can
identify the bottleneck lane and report realized throughput per unit.

Usage:
  python tpu_roofline.py <llo_dir> --bytes <N>
  python tpu_roofline.py <llo_dir> --inputs bf16:8192x8192,bf16:8192x8192 \\
                                   --outputs bf16:8192x8192

Known modeling gaps / TODOs:

  - Phi / loop-carried accumulator stalls are not modeled. Kernels with
    a tight loop-carried dependency (e.g. `acc = acc + matmul(...)` over
    K-tiles, standalone bmm) under-predict by ~10-15% of compute time
    because the model treats each iteration as independent. Attention's
    bmm-inside-fusion predicts fine because softmax/mask work between
    matmuls hides the dependency. Verified on bmm 8x1024x1024:
    un-autotuned -17.7%, autotuned (43x faster overall) still -11.8%.
    So the gap is structural, not a config artifact, though the absolute
    µs error scales with kernel time (472 µs un-autotuned, 7 µs autotuned).

  - The `--cycle-accurate` operand-graph simulator is experimental and
    currently *worse* than the default for most kernels (over-predicts
    attention by +28-45%) because it stacks dependency stalls on top of
    the lane-realization stretch the default model already applies, and
    its per-op latency table is guessed rather than measured. Promoting
    it would require (1) dropping per-lane realization factors and
    re-calibrating purely from operand latencies, and (2) microbenchmark
    each TPU v7x instruction (vmatmul, vexp, vload, vstore, etc.) to
    derive real latencies. Multi-day project; only worth it if (a)
    above turns out to need phi tracking.

  - GMM at larger M (e.g. M=4096): outer-M tile count is not in
    `loop_factor` and `--inputs` undercounts HBM bytes when RHS is
    re-read across N-tiles. Compounds to ~50% under-prediction.
    Workaround: back-solve via `scripts/hlo_sidecar.py back-solve` and
    persist K. Structural fix would require (a) parsing the kernel-name
    suffix (`tm_NN-tk_NN-tn_NN`) for tile sizes and combining with a
    user-supplied `--shape` to derive K, AND (b) a `--bytes-multiplier`
    or DMA-pattern scanner for re-read accounting.

How to produce the LLO dump that this script consumes:
  # On the TPU pod (Linux, libtpu installed):
  rm -rf /tmp/llo_dump && mkdir -p /tmp/llo_dump
  LIBTPU_INIT_ARGS='--xla_jf_dump_to=/tmp/llo_dump' \\
      ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_VISIBLE_CHIPS=<N> \\
      python <your-jax-script-or-runner>.py
  # Then identify the kernel of interest among the dumps:
  ls -laS /tmp/llo_dump/*final_bundles.txt | head    # largest = your kernel
  # For Helion-on-Pallas kernels, look for `custom_kernel.<N>-final_bundles.txt`;
  # for msl-tpu-kernel kernels the name is descriptive (e.g. `gmm_v2-g_8-...`).
  # Copy both the bundle file and its companion utilization file:
  cp /tmp/llo_dump/<kernel>-...-final_bundles.txt                    <entry>/final_bundles.txt
  cp /tmp/llo_dump/<kernel>-...-final_hlo-static-per-bundle-utilization.txt <entry>/utilization.txt
  # Then predict:
  python scripts/tpu_roofline.py <entry> --inputs ... --outputs ... --measured-us X
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import operator
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
# pyrefly: ignore[missing-import]  # llo_parse.py lives next to this file
from llo_parse import parse_llo_dump

# TPU v7x (Ironwood) constants — 1 TensorCore. These are hardware-specific:
# clocks, peak FLOPS, HBM bandwidth, and the per-lane realization factors below
# were all calibrated against TPU v7x. Running on a different generation
# (v5p, v6e, etc.) without re-calibration will produce wrong predictions.
SUPPORTED_TPU_VERSION = "v7x"
CLOCK_GHZ = 2.2
HBM_PEAK_GBPS = 3_690.0
HBM_EFFECTIVE_GBPS = 3_160.0  # ~85.6% of peak (measured, jnp.add streaming)
BF16_PEAK_TFLOPS = 1_153.0
BF16_EFFECTIVE_TFLOPS = 912.0  # ~79% of peak (measured, jnp.matmul 4k+/8k+)
MXU_REALIZATION = BF16_EFFECTIVE_TFLOPS / BF16_PEAK_TFLOPS  # 0.791
FLOPS_PER_VMATMUL = (
    2_097_152  # 2^21 — measured (msl-tpu-kernel README's 10485 is a typo)
)

# Per-lane effective peaks (per TC, bf16), calibrated via async-chained
# jit'd kernels with high arithmetic intensity to escape memory bound.
VALU_EFFECTIVE_TFLOPS = 11.3  # mul/add chain, K=64 saturated
TRANS_EFFECTIVE_TFLOPS = 3.87  # transcendentals (exp/sin/cos/...) via VALU+EUP
MXU_UNITS = 2
MXU_ISSUE_PERIOD = 4  # cycles between consecutive vmatmul issues

# Per-lane pipeline-realization factors: real hardware sustains only this
# fraction of the schedule's idealized throughput when the lane is active.
# Applied as: inflation = (1 - busy) + busy / realization, per lane, summed
# under the assumption that lanes are mostly mutually exclusive within a
# bundle for the rare lanes we calibrate.
# Calibrated from:
#   MXU: jnp.matmul 8192³ (912/1153 TFLOPS = 0.791)
#   XLU: Pallas broadcast 1×1 scratch (back-fit from measured 1153µs)
LANE_REALIZATION = {
    "MXU": 0.81,  # jointly calibrated with VSTORE — previously 0.79 absorbed VSTORE too
    "XLU": 0.82,  # joint fit from bcast 1×1 + 1×128
    "VSTORE": 0.83,  # back-fit from bcast 1×128 (VSTORE-bound)
    # Other lanes default to 1.0 (not calibrated).
}
# Compute floor uses a gated-dispatch VLIW simulator:
#   - Walk bundles in order, track per-unit ready time
#   - For each bundle, dispatch_time = max(current_cycle, max(unit_ready[U]
#     for U in this bundle's active calibrated lanes))
#   - On dispatch, each active lane's ready time advances by 1/realization
#   - current_cycle = dispatch_time + 1
# This captures VLIW bundle atomicity (all needed units must be ready
# before a bundle issues) AND lets idle units cool down during bundles
# that don't use them. Architecturally principled — no blend coefficient,
# no per-bundle uniform stretch.
# Per-call overhead model:
#   predicted = max(MIN_TIME_FLOOR, ADDITIVE_OVERHEAD + max(compute, memory))
# Calibrated against async-chained jnp.add at 1k/4k/8k/16k:
#   - tiny shapes hit MIN_TIME_FLOOR (~30 µs minimum launch+DMA setup cost),
#   - large shapes converge to ADDITIVE_OVERHEAD + memory_floor.
MIN_TIME_FLOOR_US = 30.0
ADDITIVE_OVERHEAD_US = 6.0

DTYPE_BYTES = {
    "bf16": 2,
    "f16": 2,
    "fp16": 2,
    "f32": 4,
    "fp32": 4,
    "f8": 1,
    "fp8": 1,
    "i32": 4,
    "i64": 8,
    "i8": 1,
    "u8": 1,
}


@dataclass
class LloStats:
    static_bundles: int
    dynamic_bundles: int
    lane_names: list[str]
    lane_capacity: list[int]
    lane_static_busy: list[int]  # sum over static bundles of slot usage
    lane_dynamic_busy: list[int]  # scaled to dynamic bundles (loop ×iters)
    loop_factor: float  # dynamic_bundles / static_bundles
    dma_bytes_per_bundle: dict[int, int]  # {body bundle index: bytes issued there}
    util_rows: list[list[int]]  # per-bundle, per-lane slot counts (static body)


def parse_utilization(util_path: Path) -> tuple[list[str], list[int], list[list[int]]]:
    """Parse a *-final_hlo-static-per-bundle-utilization.txt file."""
    text = util_path.read_text().splitlines()
    # The compiler dump uses "CAPACTIY" (misspelled). Tolerate both.
    if not text or ("CAPACITY" not in text[0] and "CAPACTIY" not in text[0]):
        raise ValueError(f"{util_path}: not a utilization file (no CAPACITY header)")
    lane_names = [s.strip() for s in text[1].split(",")]
    capacities = [int(s) for s in text[2].split()]
    if "UTILIZATION" not in text[3]:
        raise ValueError(f"{util_path}: expected UTILIZATION marker on line 4")
    rows: list[list[int]] = []
    for line in text[4:]:
        line = line.strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) != len(lane_names):
            continue  # skip malformed
        rows.append([int(v) for v in vals])
    return lane_names, capacities, rows


def find_dumps(llo_dir: Path) -> tuple[Path, Path]:
    """Find the final_bundles + matching utilization files in a dump dir.

    Accepts two layouts:
      A. raw libtpu dump dir — many files with timestamps; pick the largest
         *-final_bundles.txt (excluding schedule-analysis) and matching util
      B. curated dir from llo/<entry>/ — exactly two files renamed to
         final_bundles.txt + utilization.txt
    """
    simple_bundles = llo_dir / "final_bundles.txt"
    simple_util = llo_dir / "utilization.txt"
    if simple_bundles.is_file() and simple_util.is_file():
        return simple_bundles, simple_util

    finals = sorted(
        [
            p
            for p in llo_dir.iterdir()
            if p.name.endswith("-final_bundles.txt")
            and "schedule-analysis" not in p.name
        ],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not finals:
        raise FileNotFoundError(f"No *-final_bundles.txt in {llo_dir}")
    bundles_path = finals[0]
    base = bundles_path.name.rsplit("-", 2)[0]

    def _util_pass_index(p: Path) -> int:
        m2 = re.search(r"-(\d+)-final_hlo", p.name)
        return int(m2.group(1)) if m2 else -1

    utils = sorted(
        [
            p
            for p in llo_dir.iterdir()
            if p.name.startswith(base)
            and "final_hlo-static-per-bundle-utilization" in p.name
        ],
        key=_util_pass_index,
        reverse=True,
    )
    if not utils:
        raise FileNotFoundError(f"No utilization file matching {base} in {llo_dir}")
    return bundles_path, utils[0]


_BUNDLE_RE = re.compile(r"^\s+(0x[0-9a-fA-F]+)\s+")
_LB_LEVEL_RE = re.compile(r"\sLB:\s(>+)")

# -------------------------------------------------------------------------
# Per-instruction tables for the cycle-accurate simulator
# -------------------------------------------------------------------------
# Maps mnemonic (exact or prefix) → functional unit name.
INSTR_UNIT: dict[str, str] = {
    # MXU — matrix multiply
    "vmatmul": "MXU",
    # VALU — vector ALU (arithmetic, moves, packs, type-casts)
    "vadd": "VALU",
    "vsub": "VALU",
    "vmul": "VALU",
    "vmac": "VALU",
    "vmov": "VALU",
    "vstv": "VALU",
    "vunpack": "VALU",
    "vpack": "VALU",
    "vcvt": "VALU",
    "vsel": "VALU",
    "vfill": "VALU",
    "vmin": "VALU",
    "vmax": "VALU",
    "vabs": "VALU",
    "vneg": "VALU",
    "vnot": "VALU",
    "vand": "VALU",
    "vor": "VALU",
    "vxor": "VALU",
    "vshll": "VALU",
    "vshrl": "VALU",
    "vsync": "VALU",
    "vsyncpa": "VALU",
    "vsyncadd": "VALU",
    # EUP — transcendentals
    "vexp": "EUP",
    "vexp2": "EUP",
    "vlog": "EUP",
    "vsin": "EUP",
    "vcos": "EUP",
    "vtanh": "EUP",
    "verf": "EUP",
    "vrsqrt": "EUP",
    "vrecip": "EUP",
    # VLOAD / VSTORE — VMEM movers
    "vld": "VLOAD",
    "vload": "VLOAD",
    "vst": "VSTORE",
    "vstore": "VSTORE",
    # VPOP — reductions
    "vpop_max": "VPOP",
    "vpop_min": "VPOP",
    "vpop_add": "VPOP",
    "vpop_sum": "VPOP",
    "vreduce": "VPOP",
    "vsum": "VPOP",
    # XLU — cross-lane shuffles, permutes, transpose
    "vpermute": "XLU",
    "vshfl": "XLU",
    "vrotate": "XLU",
    "vtranspose": "XLU",
    "vxpose": "XLU",
    "vxlane_add": "XLU",
    "vxlane_max": "XLU",
    "vxlane_min": "XLU",
    # DMA
    "dma": "DMA",
    # Predicate ALU
    "pand": "PRED",
    "por": "PRED",
    "pxor": "PRED",
    "pnand": "PRED",
    "pneg": "PRED",
    "pnor": "PRED",
    # Scalar ALU
    "sadd": "SALU",
    "ssub": "SALU",
    "smul": "SALU",
    "sdiv": "SALU",
    "smov": "SALU",
    "sphi": "SALU",
    "shll": "SALU",
    "sshll": "SALU",
    "sshrl": "SALU",
    "sand": "SALU",
    "sor": "SALU",
    "sxor": "SALU",
    "snot": "SALU",
    "scmp": "SALU",
    "sbr": "SALU",
    "sloop": "SALU",
    "shalt": "SALU",
    "sli": "SALU",
    "sla": "SALU",
    "sbge": "SALU",
    "sblt": "SALU",
    "sbeq": "SALU",
    "sbne": "SALU",
    "scalar_lea": "SALU",
    "int_to_ptr": "SALU",
    "sst": "SALU",
    "ssr": "SALU",
    "ssel": "SALU",
}
# Latency: cycles from instruction issue until its result is consumable.
# Values reflect COMPILER-ASSUMED latencies — small enough that operand
# stalls only fire when the compiler under-scheduled. Real hardware
# latency (which exceeds these for vmatmul, transcendentals) is captured
# via the lane realization factor instead. Without ground-truth latency
# tables this combination is approximate; the cycle-accurate path is
# experimental and not the default — use --cycle-accurate to opt in.
INSTR_LATENCY: dict[str, float] = {
    "vmatmul": 8.0,  # MXU pipeline depth — real cross-bundle stall source
    "vtranspose": 8.0,
    "vxpose": 8.0,
}
DEFAULT_LATENCY = 1.0
INSTR_ISSUE_PERIOD: dict[str, float] = {
    "vmatmul": 2.0,
}
DEFAULT_ISSUE_PERIOD = 1.0
# Cycle-accurate simulator still applies LANE_REALIZATION stretching;
# without ground-truth latencies, the realization factor captures the
# main "hardware sustained < theoretical" effect.
CYCLE_ACCURATE_USE_REALIZATION = True


@dataclass
class Instr:
    bundle: int
    mnemonic: str
    unit: str
    result: str | None
    operands: list[str]


def _classify_unit(mnemonic: str) -> str:
    if mnemonic in INSTR_UNIT:
        return INSTR_UNIT[mnemonic]
    base = mnemonic.split(".")[0]
    if base in INSTR_UNIT:
        return INSTR_UNIT[base]
    if mnemonic.startswith("dma"):
        return "DMA"
    if mnemonic.startswith("vmatmul"):
        return "MXU"
    if mnemonic.startswith("p"):
        return "PRED"
    if mnemonic.startswith("v"):
        return "VALU"
    if mnemonic.startswith("s"):
        return "SALU"
    return "UNKNOWN"


_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_BUNDLE_BLOCK_RE = re.compile(
    r"^\s*(0x[0-9a-fA-F]+).*?\{(.*?)\}", re.DOTALL | re.MULTILINE
)
_INSTR_HEAD_RE = re.compile(r"^(?:(%[\w]+)\s*=\s*)?(\S+(?:\.\S+)*)(.*)$", re.DOTALL)
_OPERAND_RE = re.compile(r"%[\w]+")


def parse_bundle_instructions(bundles_path: Path) -> list[Instr]:
    """Parse all instructions out of a *-final_bundles.txt file.

    Strategy:
      1. Strip /* ... */ comments globally (DOTALL).
      2. Match every `0xNN ... { ... }` bundle block.
      3. Within each block, split on `;;` and parse each instruction.
      4. Track ONLY vector-register operands (`%v*`) for dependency
         purposes — scalar/predicate ops are fast enough that they
         don't usually cause real stalls, and tracking them produces
         spurious stalls in loop-carried scalar phis.

    Returns instructions in original bundle order.
    """
    text = bundles_path.read_text()
    text = _COMMENT_RE.sub(" ", text)
    instrs: list[Instr] = []
    for m in _BUNDLE_BLOCK_RE.finditer(text):
        bundle_idx = int(m.group(1), 16)
        body = m.group(2)
        for piece in body.split(";;"):
            piece = piece.strip()
            if not piece:
                continue
            head = _INSTR_HEAD_RE.match(piece)
            if not head:
                continue
            result = head.group(1)
            mnemonic = head.group(2)
            rest = head.group(3)
            operands = _OPERAND_RE.findall(rest)
            tracked_result = result
            unit = _classify_unit(mnemonic)
            instrs.append(Instr(bundle_idx, mnemonic, unit, tracked_result, operands))
    instrs.sort(key=lambda i: i.bundle)
    return instrs


def simulate_cycle_accurate(
    instrs: list[Instr],
    lane_realization: dict[str, float],
) -> dict:
    """Cycle-accurate VLIW simulator with operand-graph dependency tracking.

    Per bundle (instructions grouped by bundle index):
      - dispatch_cycle = max(current_cycle,
                             max(unit_ready[i.unit] for i in bundle),
                             max(register_ready[op] for op in bundle.operands))
      - For each instruction: register_ready[result] = dispatch + latency
      - For each instruction: unit_ready[unit] = dispatch + issue_period / realization
      - current_cycle = dispatch + 1

    Returns:
      total_cycles: simulated cycle count for the static body
      stall_unit: cycles attributed to unit-readiness stalls per unit
      stall_operand: cycles attributed to operand RAW stalls (lumped — we don't
                      always know which producer's lane is to blame)
    """
    register_ready: dict[str, float] = {}
    unit_ready: dict[str, float] = {}
    stall_unit: dict[str, float] = {}
    stall_operand: float = 0.0
    current_cycle = 0.0

    # Group by bundle (instructions already sorted)
    from itertools import groupby

    for _bundle_idx, gi in groupby(instrs, key=lambda x: x.bundle):
        group = list(gi)

        # Determine dispatch constraints
        max_unit_ready, max_unit = current_cycle, None
        for inst in group:
            r = unit_ready.get(inst.unit, 0.0)
            if r > max_unit_ready:
                max_unit_ready = r
                max_unit = inst.unit

        max_op_ready = current_cycle
        for inst in group:
            for op in inst.operands:
                r = register_ready.get(op, 0.0)
                if r > max_op_ready:
                    max_op_ready = r

        dispatch = max(current_cycle, max_unit_ready, max_op_ready)

        # Attribute stall
        if dispatch > current_cycle:
            unit_stall_amt = max(0.0, max_unit_ready - current_cycle)
            op_stall_amt = max(0.0, max_op_ready - current_cycle)
            total_stall = dispatch - current_cycle
            if unit_stall_amt > op_stall_amt and max_unit:
                stall_unit[max_unit] = stall_unit.get(max_unit, 0.0) + total_stall
            else:
                stall_operand += total_stall

        # Issue all instructions in this bundle
        for inst in group:
            issue_period = INSTR_ISSUE_PERIOD.get(inst.mnemonic, DEFAULT_ISSUE_PERIOD)
            if CYCLE_ACCURATE_USE_REALIZATION:
                r = lane_realization.get(inst.unit, 1.0)
                eff_period = issue_period / max(r, 1e-9)
            else:
                eff_period = issue_period
            unit_ready[inst.unit] = dispatch + eff_period
            if inst.result:
                latency = INSTR_LATENCY.get(inst.mnemonic, DEFAULT_LATENCY)
                register_ready[inst.result] = dispatch + latency

        current_cycle = dispatch + 1.0

    return {
        "total_cycles": current_cycle,
        "stall_unit": stall_unit,
        "stall_operand": stall_operand,
    }


def _detect_lb_nesting(bundles_path: Path) -> int:
    """Return the deepest LB-nesting level seen in a bundles file.

    LB markers carry a depth suffix: `LB: >` is level 1, `LB: >>` is
    level 2, etc. emit_pipeline kernels nest the inner pipelined loop
    under the outer hl.tile loop, so depth > 1 is a signal that the
    schedule has a loop body that fires more than `loop_factor` times.
    """
    if not bundles_path or not bundles_path.is_file():
        return 0
    max_depth = 0
    for line in bundles_path.read_text().splitlines():
        m = _LB_LEVEL_RE.search(line)
        if m:
            max_depth = max(max_depth, len(m.group(1)))
    return max_depth


def parse_dma_per_bundle(bundles_path: Path) -> dict[int, int]:
    """Scan a *-final_bundles.txt and return {bundle_index: total_dma_bytes}.

    A "real" DMA is `dma.hbm_to_vmem` / `dma.vmem_to_hbm` appearing as an
    actual instruction inside `{ ... }` (i.e. before any `/* */` comment
    on the line). The same mnemonic appearing inside a `BoundsCheck`
    comment is just a reference, not an issue.

    Bytes per DMA come from the comment block that follows the
    instruction:
      window_bounds: (W0, W1)
      element_size_in_bytes: E
      → bytes = W0 × W1 × E
    """
    bytes_per_bundle: dict[int, int] = {}
    current_bundle = -1
    pending_dma_bundle: int | None = None
    pending_window_bounds: tuple[int, int] | None = None

    for line in bundles_path.read_text().splitlines():
        m = _BUNDLE_RE.match(line)
        if m:
            current_bundle = int(m.group(1), 16)

        # Detect real DMA: mnemonic appears before any '/*' on this line.
        comment_pos = line.find("/*")
        for kw in ("dma.hbm_to_vmem", "dma.vmem_to_hbm"):
            dma_pos = line.find(kw)
            if dma_pos >= 0 and (comment_pos < 0 or dma_pos < comment_pos):
                pending_dma_bundle = current_bundle
                pending_window_bounds = None
                break

        if pending_dma_bundle is not None:
            wb = re.search(r"window_bounds:\s*\((\d+),\s*(\d+)\)", line)
            if wb:
                pending_window_bounds = (int(wb.group(1)), int(wb.group(2)))
            es = re.search(r"element_size_in_bytes:\s*(\d+)", line)
            if es and pending_window_bounds is not None:
                element_size = int(es.group(1))
                nbytes = (
                    pending_window_bounds[0] * pending_window_bounds[1] * element_size
                )
                bytes_per_bundle[pending_dma_bundle] = (
                    bytes_per_bundle.get(pending_dma_bundle, 0) + nbytes
                )
                pending_dma_bundle = None
                pending_window_bounds = None

    return bytes_per_bundle


def stats_from_llo(llo_dir: Path) -> LloStats:
    bundles_path, util_path = find_dumps(llo_dir)
    lane_names, capacities, rows = parse_utilization(util_path)
    static = len(rows)
    lane_static_busy = [sum(r[i] for r in rows) for i in range(len(lane_names))]
    dma_bytes_per_bundle = parse_dma_per_bundle(bundles_path)
    # Caller passes the dynamic bundle count via --dynamic-bundles
    # (taken from `llo_tool collect` output). Default to static count.
    return LloStats(
        static_bundles=static,
        dynamic_bundles=static,  # caller overrides via --dynamic-bundles
        lane_names=lane_names,
        lane_capacity=capacities,
        lane_static_busy=lane_static_busy,
        lane_dynamic_busy=lane_static_busy,
        loop_factor=1.0,
        dma_bytes_per_bundle=dma_bytes_per_bundle,
        util_rows=rows,
    )


def parse_shape_spec(spec: str) -> int:
    """Parse 'bf16:8192x8192' → bytes for that tensor."""
    dtype, shape = spec.split(":")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype}; supported: {sorted(DTYPE_BYTES)}")
    dims = [int(d) for d in shape.split("x")]
    n = 1
    for d in dims:
        n *= d
    return n * DTYPE_BYTES[dtype]


def simulate_gated_dispatch(
    stats: LloStats,
    lane_realization: dict[str, float],
    *,
    zero_lanes: list[str] | None = None,
) -> dict:
    """Walk the static body's bundles with VLIW gated dispatch.

    For each bundle:
      - dispatch_time = max(current_cycle, max(unit_ready[L] for active L))
      - If dispatch_time > current_cycle, attribute the stall to the lane
        whose unit_ready caused it.
      - For each active calibrated lane, unit_ready[L] = dispatch + 1/realization
      - current_cycle = dispatch + 1

    Returns:
      avg_stretch: simulated cycles / static bundle count (≥ 1)
      stall_attribution: {lane: cycles} attributing stall to the lane that
        gated each bundle. Sums to (total_simulated - bundle_count).
      total_cycles_static: simulated cycles for one body iteration.
    """
    # Build the list of calibrated lanes. For VSTORE and VLOAD, fold in
    # the spill/fill companion columns — they share the same physical
    # unit's bandwidth, so a spill counts as VSTORE work, etc.
    fold_in: dict[str, list[int]] = {}
    for name, r in lane_realization.items():
        if name in stats.lane_names and r > 0:
            extras: list[int] = []
            if name == "VSTORE" and "VSTORE:SPILL" in stats.lane_names:
                extras.append(stats.lane_names.index("VSTORE:SPILL"))
            if name == "VLOAD" and "VLOAD:FILL" in stats.lane_names:
                extras.append(stats.lane_names.index("VLOAD:FILL"))
            fold_in[name] = extras

    cal_lanes: list[tuple[str, int, float, list[int]]] = []  # (name, idx, 1/r, extras)
    for name, r in lane_realization.items():
        if name in stats.lane_names and r > 0:
            cal_lanes.append(
                (name, stats.lane_names.index(name), 1.0 / r, fold_in.get(name, []))
            )

    unit_ready: dict[str, float] = {name: 0.0 for name, _, _, _ in cal_lanes}
    stall_attr: dict[str, float] = {name: 0.0 for name, _, _, _ in cal_lanes}
    current_cycle = 0.0

    # Lane indices to treat as "zero busy" for the counterfactual.
    zero_indices: set[int] = set()
    if zero_lanes:
        for name in zero_lanes:
            if name in stats.lane_names:
                zero_indices.add(stats.lane_names.index(name))

    def _is_active(row: list[int], idx: int, extras: list[int]) -> bool:
        if idx not in zero_indices and row[idx] > 0:
            return True
        return any(e not in zero_indices and row[e] > 0 for e in extras)

    for row in stats.util_rows:
        active = [
            (n, ir) for (n, i, ir, extras) in cal_lanes if _is_active(row, i, extras)
        ]
        if not active:
            current_cycle += 1.0
            continue
        # Find the lane causing the stall (largest unit_ready among active)
        gating_lane = None
        max_ready = current_cycle
        for n, _ in active:
            if unit_ready[n] > max_ready:
                max_ready = unit_ready[n]
                gating_lane = n
        dispatch = max(current_cycle, max_ready)
        if gating_lane is not None and dispatch > current_cycle:
            stall_attr[gating_lane] += dispatch - current_cycle
        for n, ir in active:
            unit_ready[n] = dispatch + ir
        current_cycle = dispatch + 1.0

    n_bundles = len(stats.util_rows)
    avg_stretch = (current_cycle / n_bundles) if n_bundles else 1.0
    return {
        "avg_stretch": avg_stretch,
        "stall_attribution": stall_attr,
        "total_cycles_static": current_cycle,
    }


def predict(
    stats: LloStats,
    bytes_moved: int,
    *,
    cycle_accurate_instrs: list[Instr] | None = None,
) -> dict:
    # Slot analysis — per-lane utilization
    lane_pct = []
    for name, cap, busy in zip(
        stats.lane_names,
        stats.lane_capacity,
        stats.lane_dynamic_busy,
        strict=True,
    ):
        capacity_total = cap * stats.dynamic_bundles
        pct = (busy / capacity_total * 100) if capacity_total else 0.0
        lane_pct.append((name, cap, busy, pct))

    # Identify binding lane (highest occupancy among non-memory lanes)
    binding_lane = max(lane_pct, key=operator.itemgetter(3))[0]

    # Compute floor — bundle count × clock period, with per-lane realization
    # adjustments. Real hardware delivers only `realization` × theoretical
    # throughput on calibrated lanes (MXU, XLU). Apply inflation to each
    # calibrated lane's busy fraction independently, summing — accurate when
    # calibrated lanes don't co-occupy bundles, which holds in practice for
    # MXU-heavy vs XLU-heavy regions.
    raw_compute_floor_us = stats.dynamic_bundles / CLOCK_GHZ / 1000

    # Diagnostic: which calibrated lanes show non-trivial busy fraction.
    realization_breakdown: list[tuple[str, float, float]] = []
    for lane_name, realization in LANE_REALIZATION.items():
        if lane_name not in stats.lane_names:
            continue
        idx = stats.lane_names.index(lane_name)
        cap_cycles = stats.lane_capacity[idx] * stats.dynamic_bundles
        busy_frac = stats.lane_dynamic_busy[idx] / cap_cycles if cap_cycles else 0.0
        if busy_frac > 0:
            realization_breakdown.append((lane_name, busy_frac, realization))

    # Compute floor: prefer cycle-accurate operand-graph simulator when
    # provided, fall back to gated-dispatch lane simulator. The cycle-
    # accurate path tracks per-register producer cycles + per-instruction
    # latency, so it catches cross-bundle RAW stalls (vmatmul → consumer
    # 8 cycles later, vexp → consumer 140+ cycles later, etc.) that the
    # lane-level simulator misses entirely.
    if cycle_accurate_instrs is not None:
        sim_ca = simulate_cycle_accurate(cycle_accurate_instrs, LANE_REALIZATION)
        body_cycles = sim_ca["total_cycles"]
        n_body = len(stats.util_rows) if stats.util_rows else 1
        avg_stretch = body_cycles / n_body
        stall_attribution_us = {
            lane: cycles * stats.loop_factor / CLOCK_GHZ / 1000
            for lane, cycles in sim_ca["stall_unit"].items()
        }
    else:
        sim = simulate_gated_dispatch(stats, LANE_REALIZATION)
        avg_stretch = sim["avg_stretch"]
        stall_attribution_us = {
            lane: cycles * stats.loop_factor / CLOCK_GHZ / 1000
            for lane, cycles in sim["stall_attribution"].items()
        }
    compute_floor_us = raw_compute_floor_us * avg_stretch

    # Total HBM traffic: when MXU is active (matmul-style kernels), the LLO
    # captures real tile reuse and should be trusted. For non-MXU kernels
    # (streaming elementwise, broadcast/reduction), the LLO parser
    # over-counts because prologue/epilogue DMAs get multiplied by the
    # body's loop_factor — user-shape bytes are more accurate.
    llo_dynamic_bytes = sum(stats.dma_bytes_per_bundle.values()) * stats.loop_factor
    user_bytes = bytes_moved
    mxu_active = (
        "MXU" in stats.lane_names
        and stats.lane_dynamic_busy[stats.lane_names.index("MXU")] > 0
    )
    if mxu_active and llo_dynamic_bytes > user_bytes * 1.5:
        # Matmul-style reuse — LLO is the truth.
        effective_bytes = int(llo_dynamic_bytes)
    else:
        # Streaming / non-reuse kernel — user-shape bytes.
        effective_bytes = user_bytes
    memory_floor_us = effective_bytes / HBM_EFFECTIVE_GBPS / 1000

    # Burst-memory check: within the static body, find the point where
    # cumulative DMA bytes most over-commit HBM bandwidth. The excess
    # forces the schedule to stall waiting for HBM.
    bw_bytes_per_cycle = HBM_EFFECTIVE_GBPS * 1e9 / (CLOCK_GHZ * 1e9)
    cum = 0
    max_deficit_bytes = 0.0
    for t in range(stats.static_bundles):
        cum += stats.dma_bytes_per_bundle.get(t, 0)
        capacity = (t + 1) * bw_bytes_per_cycle
        deficit = cum - capacity
        if deficit > max_deficit_bytes:
            max_deficit_bytes = deficit
    # Per body-iter stall (cycles); pay it once per loop iteration.
    body_burst_stall_cycles = max_deficit_bytes / bw_bytes_per_cycle
    burst_floor_us = body_burst_stall_cycles * stats.loop_factor / CLOCK_GHZ / 1000
    # Total parsed bytes / dynamic bytes (for sanity-check diagnostic).
    body_dma_bytes = sum(stats.dma_bytes_per_bundle.values())
    dynamic_dma_bytes = body_dma_bytes * stats.loop_factor

    additive_predicted_us = ADDITIVE_OVERHEAD_US + max(
        compute_floor_us, memory_floor_us, burst_floor_us
    )
    predicted_us = max(MIN_TIME_FLOOR_US, additive_predicted_us)

    # Regime classification
    if additive_predicted_us <= MIN_TIME_FLOOR_US:
        regime = "overhead-bound (small-shape min-time floor binds)"
    elif compute_floor_us >= memory_floor_us:
        regime = f"compute-bound ({binding_lane})"
    else:
        regime = "memory-bound"

    # Realized throughput per lane = (lane busy% in schedule) × (effective peak).
    # This is internally consistent with the bundle-count compute floor:
    # if the schedule has the lane busy 80% of bundles, the kernel realizes
    # 80% of that lane's peak throughput at the predicted time.
    def _busy_pct(lane_name: str) -> float:
        if lane_name not in stats.lane_names:
            return 0.0
        idx = stats.lane_names.index(lane_name)
        cap_cycles = stats.lane_capacity[idx] * stats.dynamic_bundles
        return stats.lane_dynamic_busy[idx] / cap_cycles * 100 if cap_cycles else 0.0

    mxu_busy_pct = _busy_pct("MXU")
    valu_busy_pct = _busy_pct("VALU")

    realized_mxu_tflops = mxu_busy_pct / 100 * BF16_EFFECTIVE_TFLOPS
    realized_valu_tflops = valu_busy_pct / 100 * VALU_EFFECTIVE_TFLOPS

    mxu_busy = (
        stats.lane_dynamic_busy[stats.lane_names.index("MXU")]
        if "MXU" in stats.lane_names
        else 0
    )

    realized_gbps = (
        effective_bytes / 1e9 / (predicted_us / 1e6) if predicted_us > 0 else 0.0
    )
    return {
        "raw_compute_floor_us": raw_compute_floor_us,
        "compute_floor_us": compute_floor_us,
        "realization_breakdown": realization_breakdown,
        "stall_attribution_us": stall_attribution_us,
        "memory_floor_us": memory_floor_us,
        "burst_floor_us": burst_floor_us,
        "body_dma_bytes": body_dma_bytes,
        "dynamic_dma_bytes": dynamic_dma_bytes,
        "user_bytes": user_bytes,
        "effective_bytes": effective_bytes,
        "additive_overhead_us": ADDITIVE_OVERHEAD_US,
        "min_time_floor_us": MIN_TIME_FLOOR_US,
        "additive_predicted_us": additive_predicted_us,
        "predicted_us": predicted_us,
        "regime": regime,
        "binding_lane": binding_lane,
        "lane_pct": lane_pct,
        "mxu_busy": mxu_busy,
        "realized_mxu_tflops": realized_mxu_tflops,
        "realized_valu_tflops": realized_valu_tflops,
        "realized_gbps": realized_gbps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="TPU v7x roofline predictor")
    p.add_argument(
        "llo_dir", help="Directory with *-final_bundles.txt + *-utilization.txt"
    )
    p.add_argument("--bytes", type=int, help="HBM bytes moved per kernel call")
    p.add_argument("--inputs", help="Comma-separated 'dtype:HxW' for inputs (read)")
    p.add_argument("--outputs", help="Comma-separated 'dtype:HxW' for outputs (write)")
    p.add_argument(
        "--dynamic-bundles",
        type=int,
        help="Dynamic bundle count (from `llo_tool collect`). Default: static count.",
    )
    p.add_argument(
        "--inner-loop-iters",
        type=int,
        default=1,
        help=(
            "Multiplier applied on top of --dynamic-bundles. For "
            "`pallas_loop_type=emit_pipeline` kernels, `llo_tool collect` "
            "reports only the outer hl.tile grid; the inner pipelined "
            "fori_loop iterations (S/block_n in attention) are NOT in the "
            "reported loop_factor and must be applied here. For `unroll` "
            "the inner loop is materialized into the body, so leave at 1."
        ),
    )
    p.add_argument(
        "--measured-us", type=float, help="Measured per-call µs, for comparison"
    )
    p.add_argument(
        "--what-if-realization",
        action="append",
        default=[],
        help=(
            "Counterfactual: override a lane's realization factor (e.g. "
            "'XLU=1.0'). Can be repeated. Prints what the predicted time "
            "would be under the override — useful for estimating the upper "
            "bound benefit of optimizing a particular lane."
        ),
    )
    p.add_argument(
        "--advise",
        action="store_true",
        help=(
            "Run optimization advisor: counterfactual each calibrated lane "
            "at realization=1.0 to estimate potential savings, plus emit "
            "pattern-based suggestions (e.g. XLU-bound → check broadcast)."
        ),
    )
    p.add_argument(
        "--lower-bound",
        action="store_true",
        help=(
            "Bypass the low-confidence guard and report the prediction anyway, "
            "labeled as a strict LOWER BOUND on runtime. Use when you know the "
            "kernel under-reports trip counts (dynamic loops, emit_pipeline) "
            "and want a best-case estimate. The true runtime is at least the "
            "reported number."
        ),
    )
    p.add_argument(
        "--tpu-version",
        default=SUPPORTED_TPU_VERSION,
        help=(
            f"TPU generation the dump was produced on. Only '{SUPPORTED_TPU_VERSION}' "
            "is supported — constants and per-lane realization factors are "
            "calibrated to that hardware. Pass another value at your own risk."
        ),
    )
    p.add_argument(
        "--cycle-accurate",
        action="store_true",
        help=(
            "EXPERIMENTAL: cycle-accurate operand-graph simulator (parses "
            "every instruction, tracks producer-consumer latency stalls). "
            "Without ground-truth libtpu latency tables this is more often "
            "less accurate than the default gated-dispatch model. Useful "
            "as a research scaffold; not recommended for predictions."
        ),
    )
    args = p.parse_args()

    # Auto-load inner_loop_iters from meta.yaml if the user didn't pass one.
    # This closes the refuse → measure → back-solve → persist → predict loop:
    # once `hlo_sidecar.py back-solve --persist` writes `inner_loop_iters: K`
    # to the entry's meta.yaml, future predictions of the same entry pick it
    # up automatically. Explicit --inner-loop-iters on the CLI always wins.
    if args.inner_loop_iters == 1:
        meta_path = Path(args.llo_dir) / "meta.yaml"
        if meta_path.is_file():
            for line in meta_path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("inner_loop_iters:"):
                    try:
                        value = int(stripped.split(":", 1)[1].split("#")[0].strip())
                    except ValueError:
                        continue
                    if value > 1:
                        args.inner_loop_iters = value
                        print(
                            f"ℹ Loaded inner_loop_iters={value} from "
                            f"{meta_path.name}",
                            file=sys.stderr,
                        )
                    break

    if args.tpu_version != SUPPORTED_TPU_VERSION:
        print(
            f"⚠ TPU version '{args.tpu_version}' is not the calibrated target "
            f"('{SUPPORTED_TPU_VERSION}'). CLOCK_GHZ, peak FLOPS/BW, and per-lane "
            "realization factors will be wrong. Predictions are not reliable.",
            file=sys.stderr,
        )

    stats = stats_from_llo(Path(args.llo_dir))

    # Auto-derive dynamic bundle count via our fork of llo_tool's parser
    # (scripts/llo_parse.py). Manual override via --dynamic-bundles wins.
    parsed = parse_llo_dump(Path(args.llo_dir))

    # Fail-loud guard: refuse to predict when the LLO almost certainly
    # under-reports the runtime trip count, unless the user has either
    # supplied --inner-loop-iters / --dynamic-bundles to compensate, or
    # opted into --lower-bound. The signal: deeply nested LB markers
    # (depth >= 3) AND the parser couldn't extract explicit `iter bound`
    # annotations — that pattern matches emit_pipeline / dynamic fori_loop
    # kernels (RPA, attention with dynamic mask iters). Shallow LB
    # (depth <= 2) with conf=False is benign because the body is typically
    # already expanded across all iterations (e.g. GMM with statically
    # baked tile counts), so the parsed dynamic_bundles is correct even
    # without explicit trip annotations.
    deep_unannotated_loops = (
        parsed.lb_nesting_depth >= 3 and not parsed.loop_inference_confident
    )
    user_provided_iter_hint = (
        args.dynamic_bundles is not None or args.inner_loop_iters > 1
    )

    if deep_unannotated_loops and not user_provided_iter_hint:
        reason = [
            (
                f"nested LB depth {parsed.lb_nesting_depth} with no explicit "
                f"`iter bound` annotations (parser heuristic returned trip="
                f"{parsed.inferred_trip_count}) — typical of emit_pipeline or "
                "dynamic-bound fori_loop kernels where the inner trip count is "
                "data-dependent and not derivable from LLO alone"
            )
        ]
        msg = (
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║  REFUSING TO PREDICT — low-confidence trip count         ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
            f"Reason: {'; '.join(reason)}\n\n"
            "The static LLO body alone does not determine runtime for this "
            "kernel. Options:\n"
            "  1. Pass --inner-loop-iters K, where K is the runtime trip "
            "count (S/block_n for emit_pipeline attention, total kv-blocks "
            "for RPA, etc.)\n"
            "  2. Pass --dynamic-bundles N if you know the total dynamic "
            "bundle count from an external source.\n"
            "  3. Pass --lower-bound to get a strict lower-bound estimate "
            "(true runtime is at least this number).\n"
        )
        if args.lower_bound:
            print(
                msg.replace("REFUSING TO PREDICT", "LOWER-BOUND MODE  "),
                file=sys.stderr,
            )
        else:
            print(msg, file=sys.stderr)
            sys.exit(2)

    if args.dynamic_bundles:
        total_dynamic = args.dynamic_bundles * args.inner_loop_iters
    else:
        total_dynamic = parsed.dynamic_bundles * args.inner_loop_iters

    if total_dynamic > stats.static_bundles:
        factor = total_dynamic / stats.static_bundles
        stats.dynamic_bundles = total_dynamic
        stats.lane_dynamic_busy = [int(b * factor) for b in stats.lane_static_busy]
        stats.loop_factor = factor

    if args.bytes:
        bytes_moved = args.bytes
    else:
        bytes_moved = 0
        for spec in (args.inputs or "").split(","):
            if spec.strip():
                bytes_moved += parse_shape_spec(spec.strip())
        for spec in (args.outputs or "").split(","):
            if spec.strip():
                bytes_moved += parse_shape_spec(spec.strip())
        if not bytes_moved:
            print("error: must pass --bytes or --inputs/--outputs", file=sys.stderr)
            sys.exit(2)

    cycle_accurate_instrs: list[Instr] | None = None
    if args.cycle_accurate:
        bundles_path = (
            Path(args.llo_dir) / "final_bundles.txt"
            if (Path(args.llo_dir) / "final_bundles.txt").is_file()
            else next(Path(args.llo_dir).glob("*-final_bundles.txt"), None)
        )
        if bundles_path is None:
            print("error: --cycle-accurate needs *-final_bundles.txt", file=sys.stderr)
            sys.exit(2)
        cycle_accurate_instrs = parse_bundle_instructions(bundles_path)
        print(
            f"[cycle-accurate] EXPERIMENTAL — parsed "
            f"{len(cycle_accurate_instrs):,} instructions from {bundles_path.name}. "
            f"Without libtpu ground-truth latency tables, expect 5-15% under-"
            f"prediction; --gated-dispatch (default) is more reliable.",
            file=sys.stderr,
        )

    r = predict(stats, bytes_moved, cycle_accurate_instrs=cycle_accurate_instrs)

    print("\n=== LLO stats ===")
    print(f"Static bundles:   {stats.static_bundles:,}")
    print(
        f"Dynamic bundles:  {stats.dynamic_bundles:,}  "
        f"(loop factor {stats.loop_factor:.1f}x)"
    )
    print(f"Bytes moved:      {bytes_moved / 1e6:.2f} MB")
    print()
    print("=== Per-lane busy (dynamic) ===")
    print(f"{'lane':>14}  {'cap':>4}  {'busy':>10}  {'util%':>7}")
    for name, cap, busy, pct in r["lane_pct"]:
        marker = " ← binding" if name == r["binding_lane"] else ""
        bar = "█" * int(pct / 4)
        print(f"{name:>14}  {cap:>4}  {busy:>10,}  {pct:>6.1f}%  {bar}{marker}")
    print()
    print("=== Roofline ===")
    raw_str = f"raw {r['raw_compute_floor_us']:.2f}"
    inflation = r["compute_floor_us"] / r["raw_compute_floor_us"]
    if inflation > 1.001:
        breakdown = r.get("realization_breakdown", [])
        parts = [
            f"{name} {busy * 100:.0f}% × 1/{real:.2f}" for name, busy, real in breakdown
        ]
        adjust_note = (
            f"  × {inflation:.3f} pipeline-realization inflation ("
            + "; ".join(parts)
            + ")"
        )
    else:
        adjust_note = ""
    print(
        f"  Compute floor:    {r['compute_floor_us']:>7.2f} µs  "
        f"({stats.dynamic_bundles:,} bundles / {CLOCK_GHZ} GHz, "
        f"{raw_str}{adjust_note})"
    )
    eff_mb = r["effective_bytes"] / 1e6
    user_mb = r["user_bytes"] / 1e6
    if r["effective_bytes"] > r["user_bytes"]:
        bytes_note = (
            f"  (LLO traffic {eff_mb:.1f} MB ≫ user-shape {user_mb:.1f} MB; "
            f"using LLO — kernel re-loads data)"
        )
    else:
        bytes_note = ""
    print(
        f"  Memory floor:     {r['memory_floor_us']:>7.2f} µs  "
        f"({eff_mb:.2f} MB / {HBM_EFFECTIVE_GBPS:.0f} GB/s){bytes_note}"
    )
    print(
        f"  Burst floor:      {r['burst_floor_us']:>7.2f} µs  "
        f"(peak intra-body cumulative-byte over-commit; "
        f"parsed {r['dynamic_dma_bytes'] / 1e6:.1f} MB DMA "
        f"across {stats.loop_factor:.1f}× body)"
    )
    print(
        f"  Additive ovhd:    {r['additive_overhead_us']:>7.2f} µs  "
        f"(added on top of max(compute, memory))"
    )
    print(
        f"  Min-time floor:   {r['min_time_floor_us']:>7.2f} µs  "
        f"(per-call minimum at small shapes)"
    )
    print("  ─────────────────────────")
    additive_str = f"{r['additive_predicted_us']:.2f}"
    label = "Lower bound:" if args.lower_bound else "Predicted:   "
    print(
        f"  {label}     {r['predicted_us']:>7.2f} µs   "
        f"← max(min_floor={MIN_TIME_FLOOR_US:.0f}, "
        f"additive={additive_str})"
    )
    if args.lower_bound:
        print(
            "                              "
            "(LOWER BOUND — true runtime is at least this; "
            "missing dynamic iters not modeled)"
        )
    print(f"  Regime:           {r['regime']}")
    print()
    print("=== Realized throughput at predicted time ===")
    mxu_pct = r["realized_mxu_tflops"] / BF16_EFFECTIVE_TFLOPS * 100
    valu_pct = r["realized_valu_tflops"] / VALU_EFFECTIVE_TFLOPS * 100
    gbps_pct = r["realized_gbps"] / HBM_EFFECTIVE_GBPS * 100
    print(
        f"  MXU:    {r['realized_mxu_tflops']:>7.1f} TFLOPS  "
        f"({mxu_pct:>5.1f}% of {BF16_EFFECTIVE_TFLOPS:.0f} effective peak)"
    )
    print(
        f"  VALU:   {r['realized_valu_tflops']:>7.2f} TFLOPS  "
        f"({valu_pct:>5.1f}% of {VALU_EFFECTIVE_TFLOPS:.1f} effective peak)"
    )
    print(
        f"  HBM:    {r['realized_gbps']:>7.1f} GB/s    "
        f"({gbps_pct:>5.1f}% of {HBM_EFFECTIVE_GBPS:.0f} effective peak)"
    )

    if args.measured_us:
        err = (r["predicted_us"] - args.measured_us) / args.measured_us * 100
        print()
        print("=== Validation ===")
        print(f"  Measured:       {args.measured_us:>7.2f} µs")
        print(f"  Predicted:      {r['predicted_us']:>7.2f} µs")
        print(f"  Error:          {err:>+6.1f}%")

    # Per-lane stall attribution (dynamic µs)
    stall = r["stall_attribution_us"]
    if any(v > 0 for v in stall.values()):
        print()
        print("=== Per-lane stall attribution ===")
        total_stall = sum(stall.values())
        for lane, us in sorted(stall.items(), key=lambda x: -x[1]):
            if us > 0:
                pct = us / r["compute_floor_us"] * 100
                pct_stall = (us / total_stall * 100) if total_stall else 0.0
                print(
                    f"  {lane:>10}: {us:>9.1f} µs stall ({pct:>4.1f}% of compute,"
                    f" {pct_stall:>5.1f}% of stall total)"
                )

    # Counterfactuals (--what-if-realization or --advise)
    if args.what_if_realization or args.advise:
        print()
        print("=== Counterfactuals ===")
        baseline = r["predicted_us"]
        if args.advise:
            # For each calibrated lane that has non-zero busy frac, set it
            # to 1.0 and rerun. This is the upper-bound benefit of fully
            # optimizing that lane.
            active_cal = [name for (name, busy, _) in r["realization_breakdown"]]
            scenarios = [(f"{n}=1.0", {n: 1.0}) for n in active_cal]
        else:
            scenarios = []
            for spec in args.what_if_realization:
                overrides: dict[str, float] = {}
                for pair in spec.split(","):
                    k, v = pair.split("=")
                    overrides[k.strip()] = float(v)
                scenarios.append((spec, overrides))

        for label, overrides in scenarios:
            new_realization = {**LANE_REALIZATION, **overrides}
            new_sim = simulate_gated_dispatch(stats, new_realization)
            new_compute = (
                stats.dynamic_bundles / CLOCK_GHZ / 1000 * new_sim["avg_stretch"]
            )
            new_pred = max(
                MIN_TIME_FLOOR_US,
                r["additive_overhead_us"]
                + max(new_compute, r["memory_floor_us"], r["burst_floor_us"]),
            )
            savings = baseline - new_pred
            pct = savings / baseline * 100
            print(
                f"  {label:>30}  → {new_pred:>9.1f} µs "
                f"(save {savings:>+8.1f} µs, {pct:>+5.1f}%)"
            )

        # Register-pressure analysis (two parts: lane-stall portion AND
        # pure-spill bundles that could be eliminated entirely).
        if args.advise:
            spill_idx = (
                stats.lane_names.index("VSTORE:SPILL")
                if "VSTORE:SPILL" in stats.lane_names
                else None
            )
            fill_idx = (
                stats.lane_names.index("VLOAD:FILL")
                if "VLOAD:FILL" in stats.lane_names
                else None
            )
            other_calibrated_idx = [
                stats.lane_names.index(name)
                for name in LANE_REALIZATION
                if name in stats.lane_names
            ]
            if spill_idx is not None or fill_idx is not None:
                # Count bundles whose ONLY calibrated activity is spill/fill.
                # These could plausibly be eliminated by reducing register
                # pressure (their work would re-pack into existing bundles).
                pure_spill_bundles = 0
                for row in stats.util_rows:
                    has_spill = (spill_idx is not None and row[spill_idx] > 0) or (
                        fill_idx is not None and row[fill_idx] > 0
                    )
                    has_other = any(row[i] > 0 for i in other_calibrated_idx)
                    if has_spill and not has_other:
                        pure_spill_bundles += 1

                # Estimate savings: pure-spill bundles × stretch × loop_factor / clock
                if pure_spill_bundles > 0:
                    body_static = len(stats.util_rows) or 1
                    spill_frac = pure_spill_bundles / body_static
                    rp_savings_us = baseline * spill_frac
                    rp_pred = baseline - rp_savings_us
                    print(
                        f"  {'-RP (kill pure-spill bundles)':>30}  → "
                        f"{rp_pred:>9.1f} µs "
                        f"(save {rp_savings_us:>+8.1f} µs, "
                        f"{spill_frac * 100:>+5.1f}%)"
                    )
                else:
                    print(
                        "  -RP: 0 pure-spill bundles found — spills are co-"
                        "located with productive work, so the lane-gating "
                        "cost of register pressure is zero. The compiler's "
                        "extra bundles (those it emitted to handle pressure) "
                        "are baked into the schedule; reducing pressure would "
                        "shorten the schedule itself (a different code change "
                        "than the simulator can model)."
                    )

    # Heuristic advisor
    if args.advise:
        print()
        print("=== Suggestions ===")
        binding = r["binding_lane"]
        # Identify the dominant stall lane (largest stall_attribution)
        dom_stall = max(stall.items(), key=operator.itemgetter(1), default=("", 0.0))

        if r["regime"].startswith("overhead-bound"):
            print(
                "  ⓘ Kernel is small enough that ~30 µs launch overhead "
                "dominates. Increase shape or amortize across more iterations "
                "for the predictor to be informative."
            )
        if r["regime"].startswith("memory-bound"):
            gbps_pct = r["realized_gbps"] / HBM_EFFECTIVE_GBPS * 100
            if gbps_pct > 80:
                print(
                    f"  ✓ Memory-bound at {r['realized_gbps']:.0f} GB/s "
                    f"({gbps_pct:.0f}% of effective HBM peak). "
                    "Already near the bandwidth ceiling — further speedup "
                    "requires reducing bytes moved (fusion, larger blocks, "
                    "data reuse)."
                )
            else:
                print(
                    f"  ⚠ Memory-bound but only at {gbps_pct:.0f}% of HBM peak. "
                    "Likely many small DMAs or strided access — try larger "
                    "block sizes for fewer, larger transfers."
                )
        if binding == "MXU" and dom_stall[0] == "MXU":
            print(
                "  ⚠ MXU is the binding lane. Levers:"
                "\n      - Larger inner-K block so the MXU pipeline amortizes setup"
                "\n      - Check matmul shapes align to 128/8 lane/sublane multiples"
                "\n      - Reduce non-MXU work between matmuls to avoid stalling MXU"
            )
        if binding == "XLU" or dom_stall[0] == "XLU":
            print(
                "  ⚠ XLU (cross-lane unit) is contributing significant stall. Levers:"
                "\n      - For Pallas: widen any (M, 1) VMEM scratch buffers to (M, 128)"
                "\n      - For implicit broadcasts in a loop: hoist via jnp.tile()"
                "\n      - Avoid non-multiples of lane/sublane (128/8) on inner dims"
                "\n        (see llo/bcast_M8192N128I1000_*/meta.yaml for the pattern)"
            )
        if binding == "VSTORE" or dom_stall[0] == "VSTORE":
            print(
                "  ⚠ VSTORE is binding. Levers:"
                "\n      - Reduce intermediate VMEM writes (keep more in registers)"
                "\n      - Check for compiler-inserted spills via VSTORE:SPILL %"
                "\n      - Smaller block_m may reduce store pressure if VMEM-bound"
            )

        # Register pressure check — independent of binding lane.
        def _lane_pct(name: str) -> float:
            if name not in stats.lane_names:
                return 0.0
            idx2 = stats.lane_names.index(name)
            cap_cyc = stats.lane_capacity[idx2] * stats.dynamic_bundles
            return stats.lane_dynamic_busy[idx2] / cap_cyc * 100 if cap_cyc else 0.0

        fill_pct = _lane_pct("VLOAD:FILL")
        spill_pct = _lane_pct("VSTORE:SPILL")
        if fill_pct > 20 or spill_pct > 20:
            print(
                f"  ⚠ Register pressure: VLOAD:FILL {fill_pct:.0f}%, "
                f"VSTORE:SPILL {spill_pct:.0f}% — compiler is spilling "
                "registers to VMEM. Levers:"
                "\n      - Reduce live ranges: compute intermediates closer to use"
                "\n      - Avoid pre-scaling tensors before a loop; apply scaling in-loop"
                "\n        (Helion PR #2373: moving Q*scale into the attention loop"
                "\n         improved TFLOPs ~2% by cutting spill traffic)"
                "\n      - For Helion: try different block sizes — fewer live tiles "
                "per iter reduces register demand"
            )
        for lane, _, _ in r["realization_breakdown"]:
            # Find lane stats
            idx = stats.lane_names.index(lane)
            pct = (
                stats.lane_dynamic_busy[idx]
                / (stats.lane_capacity[idx] * stats.dynamic_bundles)
                * 100
                if stats.dynamic_bundles
                else 0.0
            )
            if pct < 30 and lane == binding:
                print(
                    f"  ⓘ {lane} is binding but only {pct:.0f}% busy — "
                    "schedule may have gaps; check for serialization or "
                    "unbalanced loop unroll."
                )


if __name__ == "__main__":
    main()
