"""Plot MXFP8 latency by shape from a measured CSV.

The input has the columns ``series,m,n,k,us`` and is written by
``repro_mxfp8_latency_by_shape.py``. Series are drawn in file order. Shapes are
split into memory-bound and compute-bound groups from their ideal arithmetic
intensity against the B300 ridge point. A second panel shows the geomean of
every shape, and both panels share one log y-axis.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]

_ARTIFACTS = (
    Path(__file__).resolve().parents[2] / "artifacts" / "mxfp8_dense_gemm_repro"
)
DEFAULT_CSV = _ARTIFACTS / "mxfp8_latency_by_shape.csv"
DEFAULT_PNG = _ARTIFACTS / "mxfp8_latency_by_shape.png"
_COLORS = {
    "Parent Helion-Triton": "#8888A8",
    "Packed lowering Helion-Triton": "#F0A040",
    "Optimized Helion-Triton": "#1F8A65",
    "Tuned Helion-Triton": "#7B52AB",
    "Helion-CuTe": "#2E79B5",
}
_FALLBACK_COLOR = "#C0504D"
# B300 dense FP8 peak (about 4.5 PFLOP/s) over HBM bandwidth (about 8 TB/s).
_RIDGE_FLOP_PER_BYTE = 562.0
_Y_MIN = 4.0


def _intensity(m: int, n: int, k: int) -> float:
    """Ideal FLOP/byte: FP8 operands, E8M0 scales, BF16 output."""
    flops = 2 * m * n * k
    traffic = (m + n) * k + (m + n) * (k // 32) + 2 * m * n
    return flops / traffic


def read_results(
    path: Path,
) -> tuple[
    list[str], list[tuple[int, int, int]], dict[tuple[str, int, int, int], float]
]:
    """Return series names, shapes, and ``(series, m, n, k) -> us`` in file order."""
    series: list[str] = []
    shapes: list[tuple[int, int, int]] = []
    values: dict[tuple[str, int, int, int], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["series"]
            shape = (int(row["m"]), int(row["n"]), int(row["k"]))
            if name not in series:
                series.append(name)
            if shape not in shapes:
                shapes.append(shape)
            values[(name, *shape)] = float(row["us"])
    return series, shapes, values


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _short_name(name: str) -> str:
    return name.removeprefix("Helion-").removesuffix(" Helion-Triton").split()[0]


def write_latency_plot(
    csv_path: Path = DEFAULT_CSV, output: Path = DEFAULT_PNG
) -> Path:
    """Draw one grouped log-latency figure and return the PNG path."""
    series, shapes, values = read_results(csv_path)
    memory = [s for s in shapes if _intensity(*s) < _RIDGE_FLOP_PER_BYTE]
    compute = [s for s in shapes if _intensity(*s) >= _RIDGE_FLOP_PER_BYTE]
    ordered = memory + compute
    xs = [float(index) for index in range(len(memory))]
    if compute:
        xs += [len(memory) - 1 + 1.7 + index for index in range(len(compute))]
    all_values = list(values.values())
    # Leave a decade of headroom for the rotated value labels.
    y_max = 2 ** (int(max(all_values)).bit_length() + 1)

    bar_width = 0.8 / len(series)
    offsets = [
        (index - (len(series) - 1) / 2) * bar_width for index in range(len(series))
    ]
    figure, (axis, summary) = plt.subplots(
        1,
        2,
        figsize=(16.6, 7.4),
        dpi=160,
        sharey=True,
        gridspec_kw={"width_ratios": [5.6, 1.5], "wspace": 0.06},
    )
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    summary.set_facecolor("white")

    for offset, name in zip(offsets, series, strict=True):
        color = _COLORS.get(name, _FALLBACK_COLOR)
        heights = [values[(name, *shape)] for shape in ordered]
        bars = axis.bar(
            [x + offset for x in xs],
            [height - _Y_MIN for height in heights],
            bar_width * 0.92,
            bottom=_Y_MIN,
            color=color,
            zorder=3,
        )
        for bar, height in zip(bars, heights, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height * 1.06,
                f"{height:.2f}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1F1F1F",
                zorder=4,
            )

    axis.set_yscale("log")
    axis.set_ylim(_Y_MIN, y_max)
    ticks = [2**power for power in range(2, y_max.bit_length())]
    axis.set_yticks(ticks)
    axis.set_yticklabels([str(tick) for tick in ticks])
    axis.set_ylabel("Latency (µs, log scale)")
    axis.set_xticks(xs)
    axis.set_xticklabels([f"{m}×{n}\nK={k}" for m, n, k in ordered])
    axis.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
    axis.yaxis.grid(True, which="major", color="#E5E7EB", zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#D1D5DB")
    axis.tick_params(colors="#374151", length=0)

    transform = axis.get_xaxis_transform()
    groups = [("Memory-bound", xs[: len(memory)])]
    if compute:
        axis.axvline((xs[len(memory) - 1] + xs[len(memory)]) / 2, color="#D1D5DB")
        groups.append(("Compute-bound", xs[len(memory) :]))
    for label, group in groups:
        start, end = group[0] - 0.42, group[-1] + 0.42
        axis.plot(
            [start, end],
            [-0.18, -0.18],
            transform=transform,
            color="#6B7280",
            linewidth=1,
            clip_on=False,
        )
        axis.text(
            (start + end) / 2,
            -0.24,
            label,
            transform=transform,
            ha="center",
            va="top",
            fontsize=11,
            color="#111827",
            clip_on=False,
        )

    geomeans = [
        _geomean([values[(name, *shape)] for shape in shapes]) for name in series
    ]
    summary_bars = summary.bar(
        range(len(series)),
        [value - _Y_MIN for value in geomeans],
        0.72,
        bottom=_Y_MIN,
        color=[_COLORS.get(name, _FALLBACK_COLOR) for name in series],
        zorder=3,
    )
    for bar, value in zip(summary_bars, geomeans, strict=True):
        summary.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.06,
            f"{value:.2f}",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1F1F1F",
            zorder=4,
        )
    summary.set_xlim(-0.6, len(series) - 0.4)
    summary.set_xticks(range(len(series)))
    summary.set_xticklabels(
        [_short_name(name) for name in series], rotation=30, ha="right"
    )
    summary.yaxis.grid(True, which="major", color="#E5E7EB", zorder=0)
    summary.set_axisbelow(True)
    for side in ("top", "right"):
        summary.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        summary.spines[side].set_color("#D1D5DB")
    summary.tick_params(colors="#374151", length=0)
    summary.set_title("Geomean", fontsize=16, pad=28, color="#111827", loc="left")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_COLORS.get(name, _FALLBACK_COLOR))
        for name in series
    ]
    figure.legend(
        handles,
        series,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(series),
        frameon=False,
        fontsize=10,
    )
    axis.set_title("Latency by shape", fontsize=16, pad=28, color="#111827", loc="left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    print(f"wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()
    write_latency_plot(args.csv, args.output)


if __name__ == "__main__":
    main()
