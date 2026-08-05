"""Narrative matplotlib figures for the four-arm head-to-head campaign.

Two figures that the gnuplot pack in ``analyze_head_to_head.py`` does not cover,
both read from that script's ``analysis/per_kernel_arm.csv``:

  12_heatmap_latency_vs_llm     per-kernel latency relative to the LLM baseline
  13_family_walltime_vs_latency cost of safety: readiness wall time vs latency,
                                aggregated by kernel family

Run after ``analyze_head_to_head.py`` has populated the campaign::

    PYTHONPATH=scripts/helion_rag .venv/bin/python \\
      scripts/helion_rag/plot_narrative_figures.py \\
      --campaign .helion-rag/head_to_head_4arm

Needs the optional plotting extra (``pip install -e 'scripts/helion_rag[figures]'``);
without matplotlib the script exits cleanly with a note, the same way the gnuplot
figures skip when gnuplot is absent.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from pathlib import Path
import re
import sys
from collections import defaultdict
from collections.abc import Sequence

ARMS = ("lfbo", "llm", "hybrid_lfbo_llm", "contextual_rag_llm")
LABELS = {
    "lfbo": "LFBO",
    "llm": "LLM",
    "hybrid_lfbo_llm": "Hybrid",
    "contextual_rag_llm": "RAG-LLM",
}
# Validated for normal vision and CVD in light mode (all-pairs minimum
# normal-vision dE 16.3, CVD dE 9.2).
COLOR = {
    "lfbo": "#2a78d6",
    "llm": "#eb6834",
    "hybrid_lfbo_llm": "#1baf7a",
    "contextual_rag_llm": "#4a3aa7",
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BASELINE_ARM = "llm"


def family_of(workload: str) -> str:
    """Kernel family = the workload id up to its first shape separator."""
    return workload.split("-", 1)[0]


def short_label(workload: str) -> str:
    """Compact per-kernel tick label: the dimension that varies within a family."""
    shape = workload.split("-", 1)[1] if "-" in workload else workload
    if workload.startswith("matmul_split_k"):
        return f"K={shape.split('x')[1]}"
    if workload.startswith("matmul-"):
        return f"{shape.split('x')[0]}³"
    if workload.startswith(("attention-", "fp8_attention-")):
        return f"S={shape.split('x')[2]}"
    if workload.startswith("grouped_gemm-"):
        parts = re.match(r"g(\d+)m(\d+)", shape)
        if parts is not None:
            return f"g{parts.group(1)}·m{parts.group(2)}"
    if workload.startswith("swiglu-"):
        return f"{shape.split('x')[0]}²"
    if workload.startswith(("softmax-", "rms_norm-")):
        return f"N={shape.split('x')[1]}"
    if workload.startswith("rope-"):
        return f"S={shape.split('x')[3]}"
    if workload.startswith(("gdn_fwd_h-", "mamba2_chunk_scan-", "mamba2_chunk_state-")):
        sequence = re.search(r"s(\d+)", shape)
        if sequence is not None:
            return f"s={sequence.group(1)}"
    return shape


@dataclasses.dataclass(frozen=True)
class Kernel:
    """One workload's paired outcome across all four arms."""

    workload: str
    family: str
    short: str
    latency: dict[str, float]
    wall: dict[str, float]


def load_kernels(campaign: Path) -> list[Kernel]:
    """One row per workload with each arm's latency and readiness time.

    Workloads missing any arm are dropped: every figure here is a paired
    comparison, so a partial row would silently skew a family aggregate.
    """
    path = campaign / "analysis" / "per_kernel_arm.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run analyze_head_to_head.py first")
    latency: dict[str, dict[str, float]] = defaultdict(dict)
    wall: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            workload, arm = row["workload"], row["arm"]
            try:
                selected = float(row["median_selected_latency_ms"])
                readiness = float(row["median_readiness_seconds"])
            except (TypeError, ValueError):
                continue
            if selected > 0.0 and readiness > 0.0:
                latency[workload][arm] = selected
                wall[workload][arm] = readiness
    kernels = [
        Kernel(
            workload=workload,
            family=family_of(workload),
            short=short_label(workload),
            latency=latency[workload],
            wall=wall[workload],
        )
        for workload in sorted(latency)
        if all(arm in latency[workload] for arm in ARMS)
    ]
    kernels.sort(key=lambda row: (row.family, row.workload))
    return kernels


def _families(kernels: Sequence[Kernel]) -> list[str]:
    return list(dict.fromkeys(row.family for row in kernels))


def _style_axes(ax, grid_axis: str | None = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=0)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def figure_heatmap(kernels: Sequence[Kernel], out: Path) -> list[Path]:
    """Per-kernel latency relative to the LLM baseline, grouped by family."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import Normalize
    from matplotlib.transforms import blended_transform_factory

    count = len(kernels)
    ratio = np.array(
        [
            [row.latency[arm] / row.latency[BASELINE_ARM] for arm in ARMS]
            for row in kernels
        ]
    )
    values = np.log10(ratio)
    limit = math.log10(3.2)
    cmap = LinearSegmentedColormap.from_list(
        "green_red",
        [
            (0.00, "#0a7d38"),
            (0.25, "#7dc79a"),
            (0.50, "#eeede9"),
            (0.75, "#e39aa0"),
            (1.00, "#c62f2f"),
        ],
    )
    norm = Normalize(-limit, limit)

    figure, ax = plt.subplots(figsize=(7.9, max(4.0, 0.38 * count)))
    figure.subplots_adjust(left=0.30, right=0.985, top=0.905, bottom=0.075)
    mesh = ax.pcolormesh(
        np.arange(len(ARMS) + 1),
        np.arange(count + 1),
        values,
        cmap=cmap,
        norm=norm,
        edgecolors=SURFACE,
        linewidth=1.6,
    )
    ax.set_ylim(count, 0)
    ax.set_xlim(0, len(ARMS))

    for i in range(count):
        for j, arm in enumerate(ARMS):
            value = ratio[i, j]
            shade = norm(values[i, j])
            text = (
                "1.00"
                if arm == BASELINE_ARM
                else (f"{value:.2f}" if value < 10 else f"{value:.0f}")
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=7.6,
                color="#ffffff" if abs(shade - 0.5) > 0.34 else INK,
                fontweight="normal" if arm == BASELINE_ARM else "bold",
            )

    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(np.arange(len(ARMS)) + 0.5)
    ax.set_xticklabels(
        [
            f"{LABELS[arm]}\n(baseline)" if arm == BASELINE_ARM else LABELS[arm]
            for arm in ARMS
        ],
        fontsize=10.5,
        fontweight="bold",
    )
    for label, arm in zip(ax.get_xticklabels(), ARMS):
        label.set_color(COLOR[arm])
    ax.tick_params(length=0)
    ax.set_yticks(np.arange(count) + 0.5)
    ax.set_yticklabels([row.short for row in kernels], fontsize=8.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    start = 0
    for family in _families(kernels):
        size = sum(1 for row in kernels if row.family == family)
        end = start + size
        ax.plot(
            [-0.125, -0.125],
            [start + 0.14, end - 0.14],
            transform=transform,
            color=MUTED,
            lw=1.6,
            clip_on=False,
            solid_capstyle="round",
        )
        ax.text(
            -0.145,
            (start + end) / 2,
            family,
            transform=transform,
            ha="right",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=INK,
        )
        if end < count:
            ax.axhline(end, color="#ffffff", lw=3.2, zorder=5)
        start = end

    figure.text(
        0.035,
        0.966,
        "Where learned guidance beats the pure-LLM tuner",
        fontsize=14.5,
        fontweight="bold",
        ha="left",
        color=INK,
    )
    figure.text(
        0.035,
        0.940,
        "Selected kernel latency relative to the LLM baseline (LLM = 1.00).  "
        "Greener = faster, redder = slower.",
        fontsize=9.3,
        ha="left",
        color=INK2,
    )

    cax = figure.add_axes([0.30, 0.040, 0.50, 0.016])
    bar = figure.colorbar(mesh, cax=cax, orientation="horizontal")
    bar.set_ticks([math.log10(t) for t in (1 / 3, 0.5, 1, 2, 3)])
    bar.set_ticklabels(["0.33x", "0.5x", "1x (LLM)", "2x", "3x"])
    bar.ax.tick_params(labelsize=8.5, length=0, labelcolor=INK2)
    bar.outline.set_visible(False)
    cax.set_title(
        "<- faster than LLM          slower than LLM ->",
        fontsize=8.5,
        color=MUTED,
        pad=5,
    )

    written = _save(figure, out, "12_heatmap_latency_vs_llm")
    plt.close(figure)
    return written


def figure_family_bars(kernels: Sequence[Kernel], out: Path) -> list[Path]:
    """Total readiness time and latency-vs-LLM per kernel family."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    families = _families(kernels)
    total_wall = {arm: [] for arm in ARMS}
    relative_latency = {arm: [] for arm in ARMS}
    for family in families:
        members = [row for row in kernels if row.family == family]
        for arm in ARMS:
            total_wall[arm].append(sum(row.wall[arm] for row in members))
            relative_latency[arm].append(
                math.exp(
                    sum(
                        math.log(row.latency[arm] / row.latency[BASELINE_ARM])
                        for row in members
                    )
                    / len(members)
                )
            )

    x = np.arange(len(families))
    width = 0.19
    offset = {arm: (i - 1.5) * width for i, arm in enumerate(ARMS)}

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(max(8.0, 1.2 * len(families)), 8.4), sharex=True
    )
    figure.subplots_adjust(
        left=0.075, right=0.985, top=0.855, bottom=0.135, hspace=0.16
    )

    for arm in ARMS:
        top.bar(
            x + offset[arm],
            total_wall[arm],
            width * 0.9,
            color=COLOR[arm],
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=3,
        )
    for index in range(len(families)):
        hybrid = total_wall["hybrid_lfbo_llm"][index]
        top.text(
            index + offset["hybrid_lfbo_llm"],
            hybrid,
            f"{hybrid / total_wall[BASELINE_ARM][index]:.1f}x",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=COLOR["hybrid_lfbo_llm"],
            fontweight="bold",
        )
    _style_axes(top, "y")
    top.set_ylabel("Total readiness wall time (s)", fontsize=10.5, color=INK2)
    top.set_ylim(0, max(max(values) for values in total_wall.values()) * 1.14)
    top.margins(x=0.01)

    for arm in ARMS:
        bottom.bar(
            x + offset[arm],
            relative_latency[arm],
            width * 0.9,
            color=COLOR[arm],
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=3,
        )
    bottom.axhline(1.0, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
    bottom.text(
        len(families) - 0.48,
        1.0,
        "LLM baseline",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    _style_axes(bottom, "y")
    bottom.set_ylabel(
        "Selected latency vs LLM\n(per-family geomean, LLM = 1.0)",
        fontsize=10.5,
        color=INK2,
    )
    ceiling = 1.6
    bottom.set_ylim(0, ceiling)
    # Annotate rather than silently clip the bars that run off scale.
    for index in range(len(families)):
        for arm in ARMS:
            value = relative_latency[arm][index]
            if value > ceiling:
                bottom.text(
                    index + offset[arm],
                    ceiling * 0.985,
                    f"{value:.1f}x↑",
                    ha="center",
                    va="top",
                    fontsize=7.4,
                    color=COLOR[arm],
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "fc": "#ffffff",
                        "ec": "none",
                        "alpha": 0.75,
                    },
                )
    bottom.set_xticks(x)
    bottom.set_xticklabels(families, rotation=28, ha="right", fontsize=9.6, color=INK)

    top.text(
        0.0,
        1.045,
        "The cost of safety  -  tuning time grows, latency barely moves",
        transform=top.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    bottom.text(
        0.0,
        1.03,
        "Below 1.0 = faster kernel than LLM; most families sit near parity",
        transform=bottom.transAxes,
        fontsize=10,
        color=INK2,
    )
    figure.text(
        0.075,
        0.965,
        "Hybrid vs. RAG-LLM: what the extra tuning time buys",
        fontsize=15.5,
        fontweight="bold",
        ha="left",
        color=INK,
    )
    figure.text(
        0.075,
        0.938,
        "Hybrid buys small latency gains with 2-3x the tuning wall time; "
        f"RAG-LLM keeps LLM-class speed.  Aggregated over {len(kernels)} kernels "
        "by family.",
        fontsize=9.6,
        ha="left",
        color=INK2,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            ls="",
            ms=10,
            color=COLOR[arm],
            markeredgecolor="none",
            label=LABELS[arm],
        )
        for arm in ARMS
    ]
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.99),
        ncol=len(ARMS),
        frameon=False,
        fontsize=10,
        handletextpad=0.4,
        columnspacing=1.3,
    )

    written = _save(figure, out, "13_family_walltime_vs_latency")
    plt.close(figure)
    return written


def _save(figure, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "svg"):
        path = out / f"{stem}.{suffix}"
        figure.savefig(path, dpi=190)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="campaign directory analyzed by analyze_head_to_head.py",
    )
    args = parser.parse_args(argv)

    try:
        import matplotlib
    except ImportError:
        print(
            "matplotlib not found; narrative figures not rendered "
            "(install with: pip install -e 'scripts/helion_rag[figures]')",
            file=sys.stderr,
        )
        return 0
    matplotlib.use("Agg")

    kernels = load_kernels(args.campaign)
    if not kernels:
        print("no workload has all four arms; nothing to plot", file=sys.stderr)
        return 0

    figures = args.campaign / "figures"
    written = figure_heatmap(kernels, figures) + figure_family_bars(kernels, figures)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
