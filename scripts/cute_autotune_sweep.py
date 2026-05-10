"""CuTe-backend autotune sweep harness (G3.0 acceptance, cute_plan.md §7.0).

Drives a curated subset of ``test/test_examples.py`` through normal
Helion autotune under ``HELION_BACKEND=cute``. Each test runs in a
fresh subprocess with a wall-clock timeout so a hang in one example
cannot poison the next. Results are appended to a JSONL file for
later grep / diff.

Usage::

    python scripts/cute_autotune_sweep.py \\
        --output cute_sweep.jsonl

    # Run the smaller non-fp32 coverage list with a longer budget:
    python scripts/cute_autotune_sweep.py \\
        --list-name nonfp32 \\
        --output cute_sweep_nonfp32.jsonl \\
        --timeout 1800

    # Dry-run the test list (no GPU work):
    python scripts/cute_autotune_sweep.py --list

    # Run a smaller subset by substring match:
    python scripts/cute_autotune_sweep.py --filter add --filter softmax

The plugin in ``scripts/cute_autotune_sweep_plugin.py`` patches
``helion._testing.code_and_output`` / ``output_only`` so each invocation
runs ``bound.autotune(args, force=True)``. Tests that build their own
``helion.Config(...)`` directly (e.g. ``test_split_k_barrier_accuracy``,
``test_matmul_bwd``, ``test_addmm_bwd``) bypass the plugin and run with
their forced config rather than autotune. The curated node-ID list
below intentionally omits those tests; if you add new ones, drop tests
that build their own ``Config`` and call the kernel directly.

``@skipIfCute`` tests are skipped at pytest collection. ``@xfailIfCute``
tests still *run* — pytest records them as ``xfail`` on failure (or
``xpassed`` on success), but the test body executes and can still
exercise / crash the GPU. The curated list keeps ``@xfailIfCute`` cases
out so the sweep does not spend wall-clock on paths the plan already
classifies as unsupported.

GPU policy: this harness is intended for ``CUDA_VISIBLE_DEVICES=6``;
the user's environment must set that. The harness does not unset or
broaden ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_MODULE = "scripts.cute_autotune_sweep_plugin"

# Curated list of CuTe-capable example tests. The prompt says "Start
# from ``test/test_examples.py`` and ``benchmarks/run.py``'s example
# map; target at least 20 CuTe-capable examples". Selection rules:
#   * Use only tests that go through ``check_example`` / ``code_and_output``
#     / ``output_only`` — the plugin can only patch those entry points,
#     so tests that build a ``helion.Config(...)`` and call the kernel
#     directly (``test_matmul_bwd``, ``test_addmm_bwd``,
#     ``test_split_k_barrier_accuracy``) are intentionally excluded;
#     they would pass with their forced config rather than autotune.
#   * Skip ``@xfailIfCute``-marked tests: ``xfailIfCute`` does *not*
#     skip execution, so those tests would still run autotune and spend
#     wall-clock on paths the plan already classifies as unsupported.
#   * Prefer matmul variants (bmm, broadcast, layernorm-fused, split-k),
#     fused-epilogue paths (§7.1), stable non-matmul kernels (softmax,
#     reductions, norm, cross-entropy, embedding, attention), and a
#     handful of harder tests (jagged, grouped GEMM) that stress CuTe
#     lowering corners.
SWEEP_NODE_IDS: tuple[str, ...] = (
    # Pure matmul variants — the load-bearing acceptance row.
    "test/test_examples.py::TestExamples::test_matmul_default",
    "test/test_examples.py::TestExamples::test_matmul",
    "test/test_examples.py::TestExamples::test_matmul_split_k",
    "test/test_examples.py::TestExamples::test_bmm",
    "test/test_examples.py::TestExamples::test_bmm_non_divisible_k",
    "test/test_examples.py::TestExamples::test_broadcast_matmul",
    "test/test_examples.py::TestExamples::test_matmul_layernorm_static_shapes",
    "test/test_examples.py::TestExamples::test_matmul_layernorm_half_dtype_multi_k_tile",
    # Element-wise / reduction kernels — stable CuTe targets.
    "test/test_examples.py::TestExamples::test_add",
    "test/test_examples.py::TestExamples::test_add_loop_order",
    "test/test_examples.py::TestExamples::test_softmax",
    "test/test_examples.py::TestExamples::test_softmax_two_pass",
    "test/test_examples.py::TestExamples::test_cross_entropy",
    "test/test_examples.py::TestExamples::test_rms_norm_fwd",
    "test/test_examples.py::TestExamples::test_layernorm_with_bias",
    "test/test_examples.py::TestExamples::test_layernorm_no_bias",
    "test/test_examples.py::TestExamples::test_sum",
    "test/test_examples.py::TestExamples::test_long_sum",
    "test/test_examples.py::TestExamples::test_exp_fwd",
    "test/test_examples.py::TestExamples::test_concat",
    "test/test_examples.py::TestExamples::test_embedding_pointers",
    # Activations / fused-epilogue chains.
    "test/test_examples.py::TestExamples::test_geglu",
    "test/test_examples.py::TestExamples::test_swiglu",
    "test/test_examples.py::TestExamples::test_jsd",
    "test/test_examples.py::TestExamples::test_kl_div",
    # Harder / corner-case kernels.
    "test/test_examples.py::TestExamples::test_attention_pointer",
    "test/test_examples.py::TestExamples::test_jagged_dense_add",
    "test/test_examples.py::TestExamples::test_jagged_mean",
    "test/test_examples.py::TestExamples::test_grouped_gemm_jagged",
)


# Non-fp32 coverage sub-sweep. Curated list of CuTe-capable nodes whose
# matmul-shaped or attention-shaped inputs are bf16 / fp16. Distinct
# from ``SWEEP_NODE_IDS`` (which is fp32-dominated) so future cycles
# can ask "does the non-fp32 autotune path still pass cleanly across
# the example suite?" without re-running the full default list.
#
# Selection rules:
#   * Tests must use ``check_example`` / ``code_and_output`` /
#     ``output_only`` so the plugin's autotune patch fires — same
#     constraint as ``SWEEP_NODE_IDS`` above.
#   * Tests must not be ``@xfailIfCute`` / ``@skipIfCute`` /
#     ``@skipIfFn(... cute)`` for the same reason as the default
#     list.
#
# ``test_matmul_bf16_tcgen05`` is the bf16 fixture added so this
# list actually fires the ``uses_tcgen05`` codegen marker. The other
# fixtures in the list use small shapes (e.g.
# ``test_bmm_non_divisible_k`` = 4x128x384x128) that do not pass
# ``matmul_ops.enforce_dot_requirements`` (M divisible by 64 and
# M >= 64 after ``update_min_block``, etc.), so they still record
# zero tcgen05 marker hits.
SWEEP_NODE_IDS_NONFP32: tuple[str, ...] = (
    # 256^3 bf16: sized just above the tcgen05 admission floor;
    # reliably fires ``uses_tcgen05`` under autotune.
    "test/test_examples.py::TestExamples::test_matmul_bf16_tcgen05",
    # Half-precision matmul / GEMM variants. ``test_bmm`` and
    # ``test_bmm_non_divisible_k`` appear in the default list too;
    # listing them here documents intent.
    "test/test_examples.py::TestExamples::test_bmm",
    "test/test_examples.py::TestExamples::test_bmm_non_divisible_k",
    "test/test_examples.py::TestExamples::test_moe_matmul_ogs",
    "test/test_examples.py::TestExamples::test_grouped_gemm_jagged",
    # Half-precision attention.
    "test/test_examples.py::TestExamples::test_jagged_hstu_attn",
)

NODE_LIST_BY_NAME: dict[str, tuple[str, ...]] = {
    "default": SWEEP_NODE_IDS,
    "nonfp32": SWEEP_NODE_IDS_NONFP32,
}


def _filter_node_ids(
    node_ids: tuple[str, ...], substrings: list[str]
) -> tuple[str, ...]:
    if not substrings:
        return node_ids
    out: list[str] = []
    for nid in node_ids:
        if any(s in nid for s in substrings):
            out.append(nid)
    return tuple(out)


def _run_one(
    node_id: str,
    output_path: Path,
    timeout_seconds: float,
    backend: str,
    extra_pytest_args: list[str],
    autotune_budget_seconds: int | None,
    autotune_max_generations: int | None,
) -> dict[str, object]:
    """Run a single test in a fresh subprocess. Returns a result-summary dict."""
    env = os.environ.copy()
    env["HELION_BACKEND"] = backend
    env["HELION_AUTOTUNE_SWEEP_RESULT_JSON"] = str(output_path)
    # Budget caps the generation loop but not the initial-population
    # benchmark; see ``cute_plan.md`` §7.0 follow-up 1.
    if autotune_budget_seconds is not None:
        env["HELION_AUTOTUNE_BUDGET_SECONDS"] = str(autotune_budget_seconds)
    if autotune_max_generations is not None:
        env["HELION_AUTOTUNE_MAX_GENERATIONS"] = str(autotune_max_generations)
    # ``CUDA_VISIBLE_DEVICES`` is intentionally inherited from the caller
    # per the GPU policy — the harness must not unset or broaden it.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        PLUGIN_MODULE,
        "--no-header",
        "-q",
        "-rN",
        "--tb=line",
        node_id,
        *extra_pytest_args,
    ]
    started = time.monotonic()
    timed_out = False
    returncode: int | None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        # ``None`` sentinel (not ``-1``) keeps the summary's "died on
        # signal / abort" tally mutually exclusive from the "timeouts"
        # tally — see ``_print_summary``.
        returncode = None
        stdout = (
            (exc.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            (exc.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
    elapsed = time.monotonic() - started
    return {
        "nodeid": node_id,
        "wall_clock_seconds": elapsed,
        "subprocess_returncode": returncode,
        "subprocess_timed_out": timed_out,
        "subprocess_stdout_tail": stdout[-2000:],
        "subprocess_stderr_tail": stderr[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="cute_sweep_results.jsonl",
        help="JSONL output path (one record per test). Created/overwritten.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help=(
            "Per-test wall-clock timeout in seconds (default: 900). The "
            "matmul-shaped autotune paths can run >10 min per node on "
            "B200 when many candidates compile cleanly — for an "
            "exhaustive sweep that disambiguates 'budget-bound' from "
            "'real hang' on the matmul-shaped paths, pass "
            "``--timeout 1800``. Documented full-sweep recipes in "
            "``cute_plan.md`` §7.0 opt into 1800."
        ),
    )
    parser.add_argument(
        "--backend",
        default="cute",
        help="HELION_BACKEND value (default: cute).",
    )
    parser.add_argument(
        "--list-name",
        default="default",
        choices=sorted(NODE_LIST_BY_NAME.keys()),
        help=(
            "Curated node-ID list to run. ``default`` is the broad "
            "29-node CuTe-capable list; ``nonfp32`` is a smaller "
            "bf16/fp16 coverage list that now includes "
            "``test_matmul_bf16_tcgen05`` (256^3 bf16) as the first "
            "fixture above the tcgen05 admission floor; the other "
            "nodes in the list use small shapes and still record zero "
            "tcgen05 marker hits — see the list's inline comment."
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help=(
            "Run only node IDs that contain this substring. Repeatable; "
            "if any substring matches, the node ID is kept."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the (filtered) node ID list and exit.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Read the JSONL file pointed at by ``--output`` and re-print the "
            "summary without running anything. The file is consumed read-only "
            "in this mode; nothing is appended or overwritten."
        ),
    )
    parser.add_argument(
        "--extra-pytest-arg",
        action="append",
        default=[],
        help="Extra args appended to each pytest invocation (repeatable).",
    )
    parser.add_argument(
        "--autotune-budget-seconds",
        type=int,
        default=None,
        help=(
            "If set, exports ``HELION_AUTOTUNE_BUDGET_SECONDS=<value>`` to "
            "each test subprocess. Caps the autotune generation loop's "
            "wall-clock. Combined with ``--autotune-max-generations 0``, "
            "this rescues ``test_bmm`` (non-fp32 list). "
            "``test_matmul_default`` and ``test_broadcast_matmul`` still "
            "time out because the initial-population benchmark phase is "
            "not budget-aware — see ``cute_plan.md`` §7.0 follow-up 1 "
            "for the deeper fix."
        ),
    )
    parser.add_argument(
        "--autotune-max-generations",
        type=int,
        default=None,
        help=(
            "If set, exports ``HELION_AUTOTUNE_MAX_GENERATIONS=<value>`` to "
            "each test subprocess. ``0`` runs only the initial population + "
            "rebenchmark + finishing phase. Combined with "
            "``--autotune-budget-seconds``, this rescues ``test_bmm`` only; "
            "``test_matmul_default`` and ``test_broadcast_matmul`` still "
            "time out — see ``cute_plan.md`` §7.0 follow-up 1."
        ),
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    selected_list = NODE_LIST_BY_NAME[args.list_name]
    node_ids = _filter_node_ids(selected_list, args.filter)

    if args.list:
        for nid in node_ids:
            print(nid)
        print(f"# {len(node_ids)} test node IDs")
        return 0

    if args.summary_only:
        return _print_summary(output_path)

    if not node_ids:
        print("No test node IDs match the requested filters.", file=sys.stderr)
        return 1

    # Fresh JSONL output — both the per-test plugin records and the
    # per-test subprocess wrapper records below are appended.
    if output_path.exists():
        output_path.unlink()
    output_path.touch()

    print(
        f"[cute_autotune_sweep] running {len(node_ids)} tests from "
        f"list={args.list_name!r}; timeout={args.timeout}s; "
        f"backend={args.backend}; output={output_path}",
        flush=True,
    )

    summary_records: list[dict[str, object]] = []
    for i, node_id in enumerate(node_ids, start=1):
        print(f"[{i}/{len(node_ids)}] {node_id}", flush=True)
        result = _run_one(
            node_id,
            output_path,
            timeout_seconds=args.timeout,
            backend=args.backend,
            extra_pytest_args=list(args.extra_pytest_arg),
            autotune_budget_seconds=args.autotune_budget_seconds,
            autotune_max_generations=args.autotune_max_generations,
        )
        summary_records.append(result)
        # Write a wrapper record so the JSONL can correlate
        # subprocess-level outcomes (timeouts, segfaults) with the
        # plugin's per-test record (which only fires for tests that
        # actually reach pytest's ``call`` phase).
        with output_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "kind": "subprocess_summary",
                        **result,
                        # Trim the embedded stdout / stderr a bit for
                        # the inline summary; the per-test record above
                        # carries its own tail.
                        "subprocess_stdout_tail": result["subprocess_stdout_tail"][
                            -512:
                        ],  # type: ignore[index]
                        "subprocess_stderr_tail": result["subprocess_stderr_tail"][
                            -512:
                        ],  # type: ignore[index]
                    }
                )
                + "\n"
            )

    return _print_summary(output_path)


def _print_summary(output_path: Path) -> int:
    if not output_path.exists():
        print(f"No sweep output at {output_path}", file=sys.stderr)
        return 1
    print(f"\n[cute_autotune_sweep] summary from {output_path}\n")
    test_records: list[dict[str, object]] = []
    subprocess_records: list[dict[str, object]] = []
    for raw_line in output_path.read_text().splitlines():
        if not raw_line.strip():
            continue
        rec = json.loads(raw_line)
        if rec.get("kind") == "subprocess_summary":
            subprocess_records.append(rec)
        else:
            test_records.append(rec)

    outcomes: Counter[str] = Counter()
    timeouts = 0
    seg_or_signal = 0
    nonzero_rc_no_call_record: list[dict[str, object]] = []
    test_record_nodeids = {str(r.get("nodeid")) for r in test_records}
    for sub in subprocess_records:
        if sub.get("subprocess_timed_out"):
            timeouts += 1
            continue
        rc = sub.get("subprocess_returncode")
        if isinstance(rc, int) and rc < 0:
            seg_or_signal += 1
            continue
        # Pytest exits non-zero on collection failures, import errors,
        # or fixture setup failures that never reach a per-test call
        # record. Treat those as harness failures so the sweep does not
        # silently exit 0 when nothing actually ran.
        if (
            isinstance(rc, int)
            and rc != 0
            and str(sub.get("nodeid")) not in test_record_nodeids
        ):
            nonzero_rc_no_call_record.append(sub)
    for rec in test_records:
        outcomes[str(rec.get("outcome", "unknown"))] += 1

    print(f"  tests reaching pytest call phase: {len(test_records)}")
    for k, v in sorted(outcomes.items()):
        print(f"    {k}: {v}")
    print(f"  subprocess timeouts: {timeouts}")
    print(f"  subprocess died on signal / abort: {seg_or_signal}")
    print(
        f"  subprocess nonzero exit without call record: "
        f"{len(nonzero_rc_no_call_record)}"
    )
    print()

    failures = [
        r for r in test_records if r.get("outcome") not in ("passed", "skipped")
    ]
    if failures:
        print("Failures (nodeid -> outcome / autotune_seconds):")
        for rec in failures:
            print(
                f"  {rec.get('nodeid')} :: {rec.get('outcome')} "
                f"(autotune_seconds={rec.get('autotune_seconds')})"
            )
        print()
    timed_out_subs = [r for r in subprocess_records if r.get("subprocess_timed_out")]
    if timed_out_subs:
        print("Subprocess timeouts (nodeid -> wall_clock_seconds):")
        for rec in timed_out_subs:
            print(f"  {rec.get('nodeid')} :: {rec.get('wall_clock_seconds'):.1f}s")
        print()
    if nonzero_rc_no_call_record:
        print(
            "Subprocess nonzero exit without per-test call record "
            "(collection / import / setup failure):"
        )
        for rec in nonzero_rc_no_call_record:
            print(
                f"  {rec.get('nodeid')} :: rc={rec.get('subprocess_returncode')} "
                f"wall={rec.get('wall_clock_seconds'):.1f}s"
            )
        print()

    # Tests where autotune ran and the canonical CuTe codegen markers
    # fired — useful sanity that the sweep actually exercised the
    # tcgen05 path on matmul-shaped examples.
    tcgen05_hits = [
        r
        for r in test_records
        if isinstance(r.get("codegen_markers"), dict)
        and bool(r["codegen_markers"].get("uses_tcgen05"))  # type: ignore[union-attr]
    ]
    print(f"  tcgen05 codegen marker hits: {len(tcgen05_hits)}")
    if failures or timeouts or seg_or_signal or nonzero_rc_no_call_record:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
