"""The frozen suite: the Helion dashboard's nightly kernels and their inputs.

Kernels are the ``kernels`` default of ``.github/workflows/benchmark_dispatch.yml``.
Rows are what the nightly feeds each kernel: TritonBench's input generator at
the pin in ``.github/ci_commit_pins/tritonbench.txt`` (``TRITONBENCH_PIN``),
``benchmarks/run.py`` ``KERNEL_MAPPINGS`` overrides (``num_inputs``, operator
args, input patches), and ``--input-sample-mode equally-spaced-k``.  Each
kernel cites its generator as ``path:line`` at the pin.

Loading this module needs neither Triton nor TritonBench.  Changing anything
here is a new ``VERSION``; results from different versions are never mixed.
The torch eager baselines come from ``examples/``, so they are pinned by the
Helion commit each run records, not by this manifest.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
from pathlib import Path
from typing import Any

VERSION = "nightly-17-v4"
TRITONBENCH_PIN = "8ef88931aec1700d9937dfbefd8fbe5e556c0361"
MAX_NUM_INPUTS = 20  # benchmarks/run.py
PROVIDERS = ("torch_eager", "torch_compile", "helion", "mlx")
BASELINES = ("torch_eager", "torch_compile", "mlx")


@dataclass(frozen=True)
class Case:
    kernel: str
    index: int  # row id in the generator's full input list
    label: str
    dtype: str
    params: dict[str, Any] = field(hash=False)


@dataclass(frozen=True)
class Kernel:
    name: str
    module: str  # benchmarks.metal.ops.<module>
    source: str
    available: int
    num_inputs: int
    rows: tuple[dict[str, Any], ...] = field(hash=False)
    dtype: str = "float32"
    # Strict verdict against the reference, per output dtype: (rtol, atol).
    tolerance: dict[str, tuple[float, float]] = field(hash=False, default_factory=dict)
    # Looser tolerances for named baselines whose own rounding is legitimate
    # but exceeds ``tolerance``, per provider and dtype.  Helion never gets one.
    baseline_tolerance: dict[str, dict[str, tuple[float, float]]] = field(
        hash=False, default_factory=dict
    )
    notes: str = ""

    @property
    def cases(self) -> tuple[Case, ...]:
        return tuple(
            Case(
                self.name, index, _label(self.rows[index]), self.dtype, self.rows[index]
            )
            for index in selected_ids(self.available, self.num_inputs)
        )


def selected_ids(available: int, num_inputs: int) -> list[int]:
    """TritonBench's ``equally-spaced-k`` input selection."""
    num_inputs = min(num_inputs, MAX_NUM_INPUTS)
    if num_inputs >= available:
        return list(range(available))
    if num_inputs == 1:
        return [0]
    step = (available - 1) / (num_inputs - 1)
    return [int(round(i * step)) for i in range(num_inputs)]


def _label(row: dict[str, Any]) -> str:
    return "(" + ", ".join(str(v) for v in row.values()) + ")"


# Per-dtype strict tolerances.  They admit the provider's own rounding of a
# correct result and reject a dropped tail, a diagonal-only write or a wrong
# reduction, all of which are O(1) errors.
FP32 = {"float32": (1e-4, 1e-5)}
FP16 = {"float16": (1e-2, 1e-5)}
BF16 = {"bfloat16": (2e-2, 1e-3)}

_ROWS_4096 = tuple({"V": 2**i} for i in range(12, 18))

KERNELS: tuple[Kernel, ...] = (
    Kernel(
        name="rms_norm",
        module="rms_norm",
        source="tritonbench/operators/rms_norm/operator.py:98-119",
        available=6,
        num_inputs=20,
        rows=tuple({"M": 2048, "H": 2**i} for i in range(10, 16)),
        tolerance=FP32,
    ),
    Kernel(
        name="rms_norm-bwd",
        module="rms_norm_bwd",
        source="tritonbench/operators/rms_norm/operator.py:98-119; run.py num_inputs=5",
        available=6,
        num_inputs=5,
        rows=tuple({"M": 2048, "H": 2**i} for i in range(10, 16)),
        tolerance={"float32": (1e-4, 1e-4)},
    ),
    Kernel(
        name="layer_norm",
        module="layer_norm",
        source="tritonbench/operators/layer_norm/operator.py:151-178",
        available=30,
        num_inputs=20,
        rows=tuple({"M": 4096, "N": 512 * i} for i in range(2, 32)),
        tolerance=FP32,
    ),
    Kernel(
        name="layer_norm-bwd",
        module="layer_norm_bwd",
        source="tritonbench/operators/layer_norm/operator.py:151-178; run.py num_inputs=10",
        available=30,
        num_inputs=10,
        rows=tuple({"M": 4096, "N": 512 * i} for i in range(2, 32)),
        tolerance={"float32": (1e-4, 1e-4)},
    ),
    Kernel(
        name="welford",
        module="welford",
        source="tritonbench/operators/welford/operator.py:19-30,125-135; run.py num_inputs=3",
        available=10,
        num_inputs=3,
        rows=tuple(
            {"S": 262144, "D": d}
            for d in (1024, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192)
        ),
        dtype="bfloat16",
        tolerance={"bfloat16": (2e-2, 1e-2)},
        # MLX rounds the normalized value to bf16 before the affine, so its
        # error scales with |weight * x_hat|, which reaches 16 (bf16 ulp 0.125).
        baseline_tolerance={"mlx": {"bfloat16": (2e-2, 0.125)}},
    ),
    Kernel(
        name="softmax",
        module="softmax",
        source="tritonbench/operators/softmax/operator.py:378-405",
        available=98,
        num_inputs=20,
        rows=tuple({"M": 4096, "N": 128 * i} for i in range(2, 100)),
        dtype="float16",
        tolerance=FP16,
    ),
    Kernel(
        name="cross_entropy",
        module="cross_entropy",
        source="tritonbench/operators/cross_entropy/operator.py:21-61",
        available=6,
        num_inputs=20,
        rows=tuple({"N": 8 * 2048, **row} for row in _ROWS_4096),
        tolerance=FP32,
    ),
    Kernel(
        name="jsd",
        module="jsd",
        source="tritonbench/operators/jsd/operator.py:63-78",
        available=6,
        num_inputs=20,
        rows=tuple({"BT": 4 * 2048, **row} for row in _ROWS_4096),
        tolerance=FP32,
    ),
    Kernel(
        name="kl_div",
        module="kl_div",
        source="tritonbench/operators/kl_div/operator.py:25-37",
        available=6,
        num_inputs=20,
        rows=tuple({"BT": 8 * 512, **row} for row in _ROWS_4096),
        tolerance=FP32,
    ),
    Kernel(
        name="rope",
        module="rope",
        source="tritonbench/operators/rope/operator.py:33-82; run.py num_inputs=5",
        available=8,
        num_inputs=5,
        rows=tuple(
            [{"hidden": 8192, "T": t} for t in (1024, 2048, 4096, 8192, 16384)]
            + [{"hidden": h, "T": 2048} for h in (512, 2048, 8192)]
        ),
        tolerance=FP32,
    ),
    Kernel(
        name="gemm",
        module="gemm",
        source="tritonbench/operators/gemm/operator.py:149-167,920-961; run.py num_inputs=8, --non-square",
        available=12,
        num_inputs=8,
        rows=tuple(
            {"M": m, "N": n, "K": k}
            for big, k in (
                (4096, 1024),
                (4096, 2048),
                (8192, 1024),
                (8192, 2048),
                (12288, 1024),
                (12288, 2048),
            )
            for m, n in ((big, k), (k, big))
        ),
        dtype="float16",
        tolerance=FP16,
        notes="B is column-major [K, N] (TritonBench's default tn layout).",
    ),
    Kernel(
        name="grouped_gemm",
        module="grouped_gemm",
        source="tritonbench/operators/grouped_gemm/operator.py:76-88,651-697; run.py num_inputs=6",
        available=4,
        num_inputs=6,
        rows=tuple({"G": 4, "N": 2**i} for i in range(10, 14)),
        dtype="bfloat16",
        tolerance=BF16,
        notes="All groups share one B, as TritonBench generates it.",
    ),
    Kernel(
        name="int4_gemm",
        module="int4_gemm",
        source="tritonbench/operators/int4_gemm/int4_gemm.py:37-53; run.py num_inputs=10",
        available=32,
        num_inputs=10,
        rows=tuple(
            {"B": b, "L": seq, "K": k, "N": n}
            for seq in (1, 4096)
            for b in (1, 4, 16, 64)
            for k, n in ((8192, 1280), (1024, 8192), (8192, 7168), (3584, 8192))
        ),
        dtype="bfloat16",
        tolerance=BF16,
        notes="Weights are int4 values in [-8, 6] packed two per int8 along K, "
        "row-major [K/2, N]; no scales.",
    ),
    Kernel(
        name="flash_attention",
        module="flash_attention",
        source="tritonbench/operators/flash_attention/generate_inputs.py:12-73; run.py d_head=128, num_inputs=6",
        available=7,
        num_inputs=6,
        rows=tuple({"B": 4, "H": 48, "S": 2**i, "D": 128} for i in range(7, 14)),
        dtype="bfloat16",
        # The dashboard's own default (benchmarks/run.py).  BF16's atol 1e-3
        # is below the rounding error of the example's own dtypes (bf16
        # q * scale reaches 4e-3; torch SDPA's max error is 3.8e-3).
        tolerance={"bfloat16": (2e-2, 1e-2)},
    ),
    Kernel(
        name="gdn_fwd_h",
        module="gdn_fwd_h",
        source="tritonbench/operators/gdn_fwd_h/operator.py:124-148,231-287",
        available=6,
        num_inputs=20,
        rows=tuple(
            {"B": b, "H": 6, "T": t, "chunk": 64, "K": 256, "V": 512}
            for b in (1, 16)
            for t in (1024, 2048, 4096)
        ),
        dtype="bfloat16",
        # The state is rounded to bf16 every chunk and reaches |S| ~ 2300
        # (bf16 ulp 16) on the long rows.  ``u - w @ bf16(S)`` cancels, so a
        # rounding that flips one way in float32 and the other in float64
        # moves dependent outputs by up to half that ulp; float32 providers
        # agree with each other but differ from the float64 oracle by up to 5.5.
        tolerance={"bfloat16": (5e-2, 8.0)},
    ),
    Kernel(
        name="mamba2_chunk_state",
        module="mamba2_chunk_state",
        source="tritonbench/operators/mamba2_chunk_state/operator.py:92-119,204-238; run.py dA_cumsum patch",
        available=6,
        num_inputs=20,
        rows=tuple(
            {"B": b, "H": 64, "G": 1, "L": seq, "chunk": 256, "P": 64, "N": 128}
            for b in (1, 64)
            for seq in (1024, 2048, 8192)
        ),
        dtype="float16",
        tolerance={"float16": (1e-2, 1e-2)},
    ),
    Kernel(
        name="mamba2_chunk_scan",
        module="mamba2_chunk_scan",
        source="tritonbench/operators/mamba2_chunk_scan/operator.py:111-138,228-281; run.py dt/dA_cumsum patch",
        available=6,
        num_inputs=20,
        rows=tuple(
            {"B": b, "H": 64, "G": 1, "L": seq, "chunk": 256, "P": 64, "N": 128}
            for b in (1, 64)
            for seq in (1024, 2048, 8192)
        ),
        dtype="float16",
        tolerance={"float16": (1e-2, 1e-2)},
    ),
)

# A fixed spread of cells measured against itself by --calibrate.
GUARD_SET: tuple[tuple[str, int], ...] = (
    ("softmax", 0),
    ("softmax", 97),
    ("rms_norm", 0),
    ("rms_norm", 5),
    ("layer_norm", 14),
    ("cross_entropy", 2),
    ("kl_div", 1),
    ("rope", 5),
    ("gemm", 2),
    ("flash_attention", 2),
)
# Measured at the start and end of a run to detect drift.
SENTINEL: tuple[str, int] = ("gemm", 2)


def kernel(name: str) -> Kernel:
    (found,) = [k for k in KERNELS if k.name == name]
    return found


def case(name: str, index: int) -> tuple[Kernel, Case]:
    k = kernel(name)
    (found,) = [c for c in k.cases if c.index == index]
    return k, found


def sentinel() -> tuple[Kernel, Case]:
    return case(*SENTINEL)


def tolerance(kernel: Kernel, provider: str, dtype: str) -> tuple[float, float]:
    """The strict (rtol, atol) for ``provider``'s output on ``kernel``."""
    return kernel.baseline_tolerance.get(provider, kernel.tolerance)[dtype]


def manifest_hash() -> str:
    """Identifies the suite: this manifest plus the input and operator code.

    The eager baselines the operators import from ``examples/`` are not
    covered; the run's Helion commit pins them.
    """
    here = Path(__file__).parent
    sources = sorted([here / "inputs.py", *(here / "ops").glob("*.py")])
    payload = {
        "version": VERSION,
        "pin": TRITONBENCH_PIN,
        "kernels": [
            {
                **{k: v for k, v in asdict(kern).items() if k != "rows"},
                "cases": [asdict(c) for c in kern.cases],
            }
            for kern in KERNELS
        ],
        "guard_set": GUARD_SET,
        "sentinel": SENTINEL,
        "sources": {
            str(path.relative_to(here)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
