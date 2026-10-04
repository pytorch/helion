from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent


def main() -> None:
    # Saved accepted ratios only; this script performs no measurements.
    data = json.loads((HERE / "data.json").read_text())
    operations = data["operations"]
    before = data["before_geomean"]
    after = data["current_geomean"]
    assert data["matched_shapes"] == 59 and len(operations) == 20
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.subplots_adjust(left=0.35, right=0.95, top=0.835, bottom=0.17)
    positions = list(range(len(operations)))
    ax.barh(
        [y - 0.18 for y in positions],
        [r["before"] for r in operations],
        height=0.33,
        color="#db9142",
        label="Original FULL search",
    )
    ax.barh(
        [y + 0.18 for y in positions],
        [r["current"] for r in operations],
        height=0.33,
        color="#178b88",
        label="Accepted historical result",
    )
    ax.set_yticks(
        positions, [r["name"] + (" *" if r["shapes"] == 2 else "") for r in operations]
    )
    ax.invert_yaxis()
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlim(0, max(1.45, max(r["current"] for r in operations) + 0.18))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    for y, row in enumerate(operations):
        for key, offset in (("before", -0.18), ("current", 0.18)):
            ax.text(
                row[key] + 0.012,
                y + offset,
                f"{row[key]:.3f}×",
                va="center",
                fontsize=9,
            )
    ax.set_xlabel("Geometric mean of baseline time / CuTe time · higher is better")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.01), frameon=False)
    fig.text(
        0.03, 0.96, "Historical CuTe hillclimb results", fontsize=21, weight="bold"
    )
    fig.text(
        0.03,
        0.922,
        f"59 matched shapes: {before:.3f}× → {after:.3f}× against the best compared baseline",
        fontsize=13,
    )
    fig.text(
        0.03,
        0.888,
        f"{after / before:.2f}× improvement in relative score · pre-cleanup combined-source refresh stopped after 2/60 cases",
        fontsize=12,
    )
    fig.text(
        0.03,
        0.105,
        "Three shapes per variant; * softmax backward uses two matched shapes because the third original run failed correctness.",
        fontsize=10,
    )
    fig.text(
        0.03,
        0.075,
        "Baselines include Triton and successful external providers, selected separately for each shape and measurement batch.",
        fontsize=10,
    )
    fig.text(
        0.03,
        0.045,
        "Before/current providers, source revisions and GPUs vary. The improvement above is not a matched aggregate runtime speedup.",
        fontsize=10,
    )
    fig.savefig(HERE / "before-after.png", dpi=160)
    fig.savefig(HERE / "before-after.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                key: data[key]
                for key in (
                    "matched_shapes",
                    "before_geomean",
                    "current_geomean",
                    "relative_score_improvement",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
