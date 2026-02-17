from __future__ import annotations

import datetime
import json
import sys


def print_autotune_metrics(metrics: list[dict[str, object]]) -> None:
    if not metrics:
        return

    from tabulate import tabulate

    headers = [
        "Kernel",
        "Input Shapes",
        "Time (s)",
        "Configs",
        "Compile Fail",
        "Accuracy Fail",
        "Generations",
        "Best Perf (ms)",
        "Configs/s",
    ]

    rows = []
    total_time = 0.0
    total_configs = 0
    total_compile_failures = 0
    total_accuracy_failures = 0
    total_generations = 0
    total_configs_per_second = 0.0
    n = len(metrics)

    for m in metrics:
        time_s = float(m.get("autotune_time", 0))
        configs = int(m.get("num_configs_tested", 0))
        compile_fail = int(m.get("num_compile_failures", 0))
        accuracy_fail = int(m.get("num_accuracy_failures", 0))
        generations = int(m.get("num_generations", 0))
        cps = float(m.get("configs_per_second", 0))

        total_time += time_s
        total_configs += configs
        total_compile_failures += compile_fail
        total_accuracy_failures += accuracy_fail
        total_generations += generations
        total_configs_per_second += cps

        rows.append(
            [
                m.get("kernel_name", ""),
                m.get("input_shapes", ""),
                f"{time_s:.2f}",
                configs,
                compile_fail,
                accuracy_fail,
                generations,
                f"{m.get('best_perf_ms', 0):.4f}",
                f"{cps:.2f}",
            ]
        )

    rows.append(
        [
            "AVERAGE",
            "",
            f"{total_time / n:.2f}",
            f"{total_configs / n:.1f}",
            f"{total_compile_failures / n:.1f}",
            f"{total_accuracy_failures / n:.1f}",
            f"{total_generations / n:.1f}",
            "",
            f"{total_configs_per_second / n:.2f}",
        ]
    )
    rows.append(
        [
            "TOTAL",
            "",
            f"{total_time:.2f}",
            total_configs,
            total_compile_failures,
            total_accuracy_failures,
            total_generations,
            "",
            "",
        ]
    )

    print("\n" + "=" * 60, file=sys.stderr)
    print("Autotune Metrics", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(tabulate(rows, headers=headers, tablefmt="simple"), file=sys.stderr)
    print(file=sys.stderr)


def export_autotune_metrics(metrics: list[dict[str, object]], path: str) -> None:
    if not metrics:
        return

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "runs": metrics,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Autotune metrics exported to: {path}", file=sys.stderr)
