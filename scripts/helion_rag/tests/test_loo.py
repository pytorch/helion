from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from helion_rag.config import Config


def _record(kernel_name: str, shape: tuple[int, ...], key: str) -> dict:
    return {
        "kernel_name": kernel_name,
        "input_shapes": repr((shape,)),
        "workload_key": key,
        "top_n": [
            {"config": {"num_warps": 4}, "median": 1.0, "config_id": f"{key}-0"},
            {"config": {"num_warps": 8}, "median": 1.1, "config_id": f"{key}-1"},
        ],
    }


def test_workload_fold_excludes_held_out_workload_and_its_sibling_keys() -> None:
    from helion_rag.loo import records_for_workload_fold

    records = [
        _record("target", (128,), "a"),
        # Same (kernel, shapes, dtypes) as "a" but a different workload_key,
        # as happens across CI snapshots with differing codegen settings.
        _record("target", (128,), "a_sibling"),
        _record("target", (256,), "b"),
        _record("other", (128,), "c"),
    ]

    fold = records_for_workload_fold(records, "a")

    # Both "a" and its sibling (same shapes+dtypes) are dropped to prevent
    # Tier-0/Tier-1 leakage; the kernel's other shape ("b") survives.
    assert [record["workload_key"] for record in fold] == ["b", "c"]


def test_select_heldout_workloads_spans_the_size_range() -> None:
    from helion_rag.loo import select_heldout_workloads

    records = [_record("target", (size,), str(size)) for size in (1, 2, 4, 8, 16)]

    chosen = select_heldout_workloads(records, "target", count=3)

    assert [record["workload_key"] for record in chosen] == ["1", "4", "16"]


def test_prepare_workload_fold_keeps_the_held_out_kernel_present(
    tmp_path: Path, monkeypatch
) -> None:
    from helion_rag.loo import prepare_workload_fold

    records = [
        {**_record("target", (128,), "a"), "family": "h100"},
        {**_record("target", (256,), "b"), "family": "h100"},
        {**_record("other", (128,), "c"), "family": "h100"},
    ]
    cfg = Config(
        embed_model="model",
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "indexes",
        writeback_dir=tmp_path / "writeback",
    )
    captured = {}

    def fake_build(fold_cfg, family, fold_records):
        captured.update(records=fold_records)
        return fold_cfg.index_dir / family / "generations" / "000000"

    # Stubbed at the module boundary rather than through a production parameter:
    # a real build would embed the corpus with the 8B model.
    monkeypatch.setattr("helion_rag.index.build_family_index", fake_build)
    fold_cfg = prepare_workload_fold(cfg, "h100", "a", records, retrieval={"k": 3})

    # Only the held-out workload is excluded; same-kernel shape "b" stays.
    assert sorted(record["workload_key"] for record in captured["records"]) == [
        "b",
        "c",
    ]
    manifest = json.loads((fold_cfg.index_dir / "fold.json").read_text())
    assert manifest["regime"] == "loo_workload"
    assert manifest["excluded_workload_key"] == "a"
    assert manifest["kernel_name"] == "target"
    # The held-out kernel type remains reachable through its other shape.
    assert "target" in manifest["included_kernels"]
    assert manifest["retrieval"] == {"k": 3}


def test_experiment_spec_provenance_in_resume_keys() -> None:
    from helion_rag.loo_experiment import build_matrix

    workloads = [
        {**_record("target", (128,), "a"), "dtypes": "('torch.float16',)"},
        {**_record("target", (256,), "b"), "dtypes": "('torch.float16',)"},
    ]
    spec = {
        "embed_model": "Qwen/Qwen3-Embedding-8B",
        "embed_text": "minimalist",
        "retrieval": {
            "sim_threshold": 0.75,
            "k": 3,
            "top_n": 3,
            "shape_rerank": True,
        },
    }

    matrix = build_matrix(
        workloads,
        repetitions=2,
        corpus_fingerprint="f" * 64,
        phase="eval",
        experiment_spec=spec,
    )

    for item in matrix:
        for key, value in spec.items():
            assert item[key] == value
    # Rebuilding with the same spec is byte-stable.
    assert matrix == build_matrix(
        workloads,
        repetitions=2,
        corpus_fingerprint="f" * 64,
        phase="eval",
        experiment_spec=spec,
    )
    # Changing only the embedding model disjoints every resume key.
    other = build_matrix(
        workloads,
        repetitions=2,
        corpus_fingerprint="f" * 64,
        phase="eval",
        experiment_spec={**spec, "embed_model": "Qwen/Qwen3-Embedding-0.6B"},
    )
    assert {item["resume_key"] for item in matrix}.isdisjoint(
        {item["resume_key"] for item in other}
    )


def test_workload_matrix_covers_both_arms_and_keys_on_phase() -> None:
    from helion_rag.loo_experiment import WORKLOAD_ARMS
    from helion_rag.loo_experiment import build_matrix

    workloads = [
        {**_record("target", (128,), "a"), "dtypes": "('torch.float16',)"},
        {**_record("target", (256,), "b"), "dtypes": "('torch.float16',)"},
    ]

    matrix = build_matrix(
        workloads,
        repetitions=2,
        corpus_fingerprint="f" * 64,
        arms=WORKLOAD_ARMS,
        phase="eval",
    )

    assert {item["arm"] for item in matrix} == {"lfbo", "rag_lfbo"}
    assert all(item["phase"] == "eval" for item in matrix)
    assert len({item["resume_key"] for item in matrix}) == len(matrix)
    # phase is part of the resume key: preflight and eval never collide.
    preflight = build_matrix(
        workloads,
        repetitions=2,
        corpus_fingerprint="f" * 64,
        arms=("lfbo",),
        phase="preflight",
    )
    assert {item["resume_key"] for item in matrix}.isdisjoint(
        {item["resume_key"] for item in preflight}
    )


def test_arm_environment_differs_only_by_the_seeding_switch(tmp_path: Path) -> None:
    from helion_rag.loo_experiment import arm_environment

    baseline = arm_environment("lfbo", tmp_path)
    candidate = arm_environment("rag_lfbo", tmp_path)

    assert baseline["HELION_RAG_LOO_SEEDING"] == "0"
    assert candidate["HELION_RAG_LOO_SEEDING"] == "1"
    # Everything else identical: the seeds are the only treatment.
    assert {k: v for k, v in baseline.items() if k != "HELION_RAG_LOO_SEEDING"} == {
        k: v for k, v in candidate.items() if k != "HELION_RAG_LOO_SEEDING"
    }
    assert candidate["HELION_AUTOTUNER"] == "LFBOTreeSearch"


def test_counterbalanced_order_pairs_are_adjacent_and_alternate() -> None:
    from helion_rag.loo_experiment import WORKLOAD_ARMS
    from helion_rag.loo_experiment import build_matrix
    from helion_rag.loo_experiment import counterbalanced_order

    workloads = [
        {**_record("target", (128,), "a"), "dtypes": "('torch.float16',)"},
        {**_record("target", (256,), "b"), "dtypes": "('torch.float16',)"},
    ]
    matrix = build_matrix(
        workloads,
        repetitions=2,
        corpus_fingerprint="f" * 64,
        arms=WORKLOAD_ARMS,
        phase="eval",
    )

    ordered = counterbalanced_order(matrix, seed=0)

    assert len(ordered) == len(matrix)
    assert [cell["run_index"] for cell in ordered] == list(range(len(ordered)))
    # Each matched pair occupies adjacent slots with the same (kernel, workload, rep).
    for start in range(0, len(ordered), 2):
        first, second = ordered[start], ordered[start + 1]
        assert (first["kernel"], first["workload_key"], first["rep"]) == (
            second["kernel"],
            second["workload_key"],
            second["rep"],
        )
        assert {first["arm"], second["arm"]} == {"lfbo", "rag_lfbo"}
    assert [cell["arm"] for cell in counterbalanced_order(matrix, seed=0)] == [
        cell["arm"] for cell in ordered
    ]
    # Counterbalancing is only worth anything if both orders actually occur;
    # if one arm always led, thermal drift would bias every pair the same way.
    assert {ordered[start]["arm"] for start in range(0, len(ordered), 2)} == {
        "lfbo",
        "rag_lfbo",
    }


def test_eligibility_excludes_kernels_below_the_min_per_kernel_bar() -> None:
    from helion_rag.loo_experiment import pair_eligible_workload_keys
    from helion_rag.loo_experiment import select_workload_folds

    records = [
        _record("k1", (size,), key) for size, key in ((1, "s1"), (4, "s2"), (16, "s3"))
    ] + [_record("k2", (size,), key) for size, key in ((2, "t1"), (8, "t2"))]
    # The baseline fails to preflight on t2, leaving k2 with one eligible shape.
    preflight = [
        {"arm": "lfbo", "workload_key": key, "ok": True}
        for key in ("s1", "s2", "s3", "t1")
    ] + [{"arm": "lfbo", "workload_key": "t2", "ok": False}]

    lfbo_ok = pair_eligible_workload_keys(preflight, "lfbo")
    assert lfbo_ok == {"s1", "s2", "s3", "t1"}

    selected, pair_kernels = select_workload_folds(
        records, {"lfbo": lfbo_ok}, count=3, min_per_kernel=2
    )

    # One fold per eligible held-out workload; the ineligible t2 is not selected.
    assert {workload["workload_key"] for workload in selected} == {
        "s1",
        "s2",
        "s3",
        "t1",
    }
    # k1 keeps 3 eligible; k2 keeps only 1 -> excluded from the analysis.
    assert pair_kernels["rag_lfbo"] == ["k1"]


def test_pending_matrix_skips_only_successful_resume_keys(tmp_path: Path) -> None:
    from helion_rag.loo_experiment import pending_matrix

    out = tmp_path / "results.jsonl"
    out.write_text(
        "\n".join(
            [
                json.dumps({"resume_key": "done", "ok": True}),
                json.dumps({"resume_key": "retry", "ok": False}),
                "truncated",
            ]
        )
        + "\n"
    )
    matrix = [
        {"resume_key": "done"},
        {"resume_key": "retry"},
        {"resume_key": "new"},
    ]

    assert [item["resume_key"] for item in pending_matrix(matrix, out)] == [
        "retry",
        "new",
    ]


def test_shape_distance_is_log_scaled_and_rejects_different_structures() -> None:
    import math

    from helion_rag.shape_distance import shape_distance

    assert shape_distance("((128, 256),)", "((128, 256),)") == 0.0
    assert shape_distance("((128, 256),)", "((256, 256),)") == 1.0
    assert math.isinf(shape_distance("((128,),)", "((128,), (128,))"))


def test_embedding_text_variants_match_between_index_and_query() -> None:
    from helion_rag.embedding_text import index_text
    from helion_rag.embedding_text import query_text

    record = {
        "embed_text": "def kernel(x):\n    return x\n",
        "kernel_name": "kernel",
        "input_shapes": "((128,),)",
        "dtypes": "('torch.float16',)",
    }
    for variant in ("source", "cleaned", "comprehensive", "minimalist"):
        assert index_text(record, variant) == query_text(
            record["embed_text"],
            record["input_shapes"],
            record["dtypes"],
            record["kernel_name"],
            variant,
        )


def test_oracle_matrix_has_one_top_five_rebench_per_workload() -> None:
    from helion_rag.loo_experiment import build_oracle_matrix

    workload = {
        **_record("target", (128,), "a"),
        "dtypes": "('torch.float16',)",
        "top_n": [
            {"config": {"num_warps": value}, "median": float(value)}
            for value in range(1, 8)
        ],
    }

    matrix = build_oracle_matrix(
        [workload], corpus_fingerprint="f" * 64, code_revision="rev"
    )

    assert len(matrix) == 1
    assert matrix[0]["arm"] == "oracle"
    assert len(matrix[0]["oracle_configs"]) == 5


def _wl_row(kernel, workload, rep, arm, **fields) -> dict:
    row = {
        "kernel": kernel,
        "workload_key": workload,
        "rep": rep,
        "arm": arm,
        "phase": "eval",
        "ok": True,
        "status": "ok",
    }
    row.update(fields)
    return row


def test_mcnemar_one_sided_p_flags_excess_candidate_failures() -> None:
    from helion_rag.loo_report import _mcnemar_one_sided_p

    assert _mcnemar_one_sided_p(0, 0) == 1.0
    assert _mcnemar_one_sided_p(5, 0) == 0.5**5  # 0.03125 -> significant excess
    assert _mcnemar_one_sided_p(1, 1) == 0.75  # symmetric -> not significant


def test_completion_table_and_gate_classify_four_outcomes() -> None:
    from helion_rag.loo_report import _completion_gate
    from helion_rag.loo_report import _completion_table

    rows = [
        _wl_row("k", "A", 0, "lfbo", perf_ms=1.0),
        _wl_row("k", "A", 0, "rag_lfbo", perf_ms=1.0),
        _wl_row("k", "B", 0, "lfbo", perf_ms=1.0),
        _wl_row("k", "B", 0, "rag_lfbo", ok=False, status="oom", perf_ms=None),
        _wl_row("k", "C", 0, "lfbo", ok=False, status="compile_fail", perf_ms=None),
        _wl_row("k", "C", 0, "rag_lfbo", perf_ms=1.0),
    ]

    table, per_workload = _completion_table(rows, "rag_lfbo", "lfbo")

    assert table == {
        "both_complete": 1,
        "baseline_complete_candidate_failed": 1,
        "baseline_failed_candidate_complete": 1,
        "both_failed": 0,
    }
    assert per_workload[("k", "B")] == "baseline_complete_candidate_failed"
    gate = _completion_gate(table)
    # candidate completes 2/3 -> below the 95% point threshold.
    assert gate["candidate_completion"] == 2 / 3
    assert gate["passes"] is False


def test_tipping_points_impute_only_candidate_only_failures() -> None:
    from helion_rag.loo_report import _tipping_points

    perf_observations = [
        {"kernel": "k", "workload_key": "A", "rep": 0, "ratio": 1.0},
    ]
    per_workload = {
        ("k", "A"): "both_complete",
        ("k", "B"): "baseline_complete_candidate_failed",
        ("k", "C"): "baseline_failed_candidate_complete",
    }

    tips = _tipping_points(perf_observations, per_workload, penalties=(2.0,))

    # A contributes log(1.0)=0; B is imputed at log(2.0); C (failed baseline) is
    # never imputed -> geomean of {1.0, 2.0} = sqrt(2).
    assert abs(tips["2.0"] - math.sqrt(2.0)) < 1e-9


def test_analyze_workload_results_verdicts_on_the_lfbo_criterion() -> None:
    from helion_rag.loo_report import analyze_workload_results

    def _rows(autotune_candidate_s: float) -> list[dict]:
        rows = []
        for kernel in ("k1", "k2", "k3"):
            for workload in ("w1", "w2"):
                for rep in (0, 1):
                    rows.append(
                        _wl_row(
                            kernel,
                            workload,
                            rep,
                            "lfbo",
                            perf_ms=2.0,
                            autotune_time_s=20.0,
                            end_to_end_s=20.0,
                        )
                    )
                    rows.append(
                        _wl_row(
                            kernel,
                            workload,
                            rep,
                            "rag_lfbo",
                            perf_ms=2.0,
                            autotune_time_s=autotune_candidate_s,
                            end_to_end_s=autotune_candidate_s,
                            tier=1,
                            same_kernel_neighbor_rate=1.0,
                        )
                    )
        return rows

    # Perf parity with the search time halved clears both halves of the rule.
    report = analyze_workload_results(_rows(10.0), [], bootstrap_samples=200)
    pair = report["pairs"]["rag_lfbo"]
    assert pair["perf_ratio"]["estimate"] == 1.0
    assert pair["autotune_ratio"]["estimate"] == 0.5
    assert pair["completion"]["passes"] is True
    assert pair["verdict"] == "effective"
    assert report["diagnostics"]["heldout_shape_leakage_count"] == 0
    assert report["diagnostics"]["tier1_coverage"] == 1.0

    # Same perf parity but no search-time win is non-inferiority only, not a win.
    slower = analyze_workload_results(_rows(20.0), [], bootstrap_samples=200)
    assert slower["pairs"]["rag_lfbo"]["verdict"] == "non_inferior_only"


def test_neighbor_limit_counts_distinct_kernel_types() -> None:
    from helion_rag.lookup import _distinct_kernel_neighbors

    neighbors = [
        {"kernel_name": "a", "input_shapes": "((1,),)"},
        {"kernel_name": "a", "input_shapes": "((2,),)"},
        {"kernel_name": "b", "input_shapes": "((1,),)"},
        {"kernel_name": "c", "input_shapes": "((1,),)"},
    ]

    selected = _distinct_kernel_neighbors(neighbors, 2)

    assert [neighbor["kernel_name"] for neighbor in selected] == ["a", "b"]


def test_curated_seed_configs_caps_and_orders_by_shape_proximity() -> None:
    from helion_rag.patch import _curated_seed_configs

    result = {
        "neighbors": [
            {
                "shape_distance": 2.0,
                "relevance": 0.5,
                "top_n": [{"config": {"num_warps": w}} for w in (1, 2, 3)],
            },
            {
                "shape_distance": 0.1,
                "relevance": 0.9,
                "top_n": [{"config": {"num_warps": w}} for w in (10, 11, 12)],
            },
            {
                "shape_distance": 1.0,
                "relevance": 0.7,
                "top_n": [{"config": {"num_warps": w}} for w in (20, 21, 22)],
            },
        ]
    }

    configs = _curated_seed_configs(result)

    # 9 available -> capped at 3; round-robin over shape-sorted neighbours
    # (near 0.1, mid 1.0, far 2.0) takes each one's best config first.
    assert [config["num_warps"] for config in configs] == [10, 20, 1]


def test_lfbo_tier1_hard_seeds_are_curated_and_capped(monkeypatch) -> None:
    import helion_rag.patch as patch

    settings = SimpleNamespace(autotune_seed_configs=None, force_autotune=False)
    bound = SimpleNamespace(_config=None, configs=[], settings=settings)
    result = {
        "tier": 1,
        "neighbors": [
            {
                "kernel_name": f"near_{i}",
                "shape_distance": float(i),
                "relevance": 1.0 / (i + 1),
                "top_n": [{"config": {"num_warps": 10 * i + r}} for r in range(3)],
            }
            for i in range(3)
        ],
    }
    seen = {}

    def original(bound_kernel, args, *rest, **kwargs):
        seen["seeds"] = list(bound_kernel.settings.autotune_seed_configs)
        return "autotuned"

    monkeypatch.setenv("HELION_RAG_LOO_SEEDING", "1")
    monkeypatch.setattr(
        patch,
        "_extract",
        lambda bound_kernel, args: {
            "kernel_name": "target",
            "kernel_source": "def target(x): return x",
            "shapes": "((128,),)",
            "dtypes": "('torch.float16',)",
            "hardware": "h100",
            "settings": {},
        },
    )
    monkeypatch.setattr(patch, "lookup", lambda *args, **kwargs: result)

    out = patch.apply(bound, original, (), (), {})

    assert out == "autotuned"
    # 9 retrieved configs are curated and capped to 3 hard seeds.
    assert len(seen["seeds"]) == 3
    # The override is scoped to the run; the kernel's own setting is restored.
    assert settings.autotune_seed_configs is None


def test_install_refuses_to_run_alongside_the_in_tree_rag_adapter(monkeypatch) -> None:
    import helion_rag.patch as patch

    monkeypatch.setenv("HELION_RAG_LOO_SEEDING", "1")
    monkeypatch.setenv("HELION_RAG_ENABLED", "1")

    # Both paths would retrieve and seed independently; fail loudly instead.
    with pytest.raises(AssertionError, match="HELION_RAG_ENABLED"):
        patch.install()


def test_neighbor_metrics_report_coverage_and_leakage_proxy() -> None:
    from helion_rag.loo_run import _neighbor_metrics

    lookup = {
        "tier": 1,
        "neighbors": [
            {"kernel_name": "target", "input_shapes": "((256,),)"},
            {"kernel_name": "target", "input_shapes": "((512,),)"},
            {"kernel_name": "other", "input_shapes": "((128,),)"},
        ],
    }

    metrics = _neighbor_metrics(
        lookup,
        tier=1,
        retrieval_s=0.5,
        target_kernel="target",
        target_shapes="((128,),)",
    )

    assert metrics["tier"] == 1
    assert metrics["retrieval_s"] == 0.5
    assert metrics["retrieved_kernels"] == ["target", "target", "other"]
    # 2 of 3 neighbours share the target kernel identity.
    assert metrics["same_kernel_neighbor_rate"] == 2 / 3
    # The held-out (target, (128,)) shape is absent -> no per-cell leakage.
    assert metrics["heldout_shape_leaked"] is False


def test_neighbor_metrics_flag_a_leaked_held_out_shape() -> None:
    from helion_rag.loo_run import _neighbor_metrics

    lookup = {"neighbors": [{"kernel_name": "target", "input_shapes": "((128,),)"}]}

    metrics = _neighbor_metrics(
        lookup,
        tier=1,
        retrieval_s=0.0,
        target_kernel="target",
        target_shapes="((128,),)",
    )

    assert metrics["heldout_shape_leaked"] is True


def test_neighbor_metrics_handle_a_retrieval_miss() -> None:
    from helion_rag.loo_run import _neighbor_metrics

    metrics = _neighbor_metrics(
        None, tier=2, retrieval_s=0.2, target_kernel="target", target_shapes="((128,),)"
    )

    assert metrics["tier"] == 2
    assert metrics["retrieval_s"] == 0.2
    assert metrics["retrieved_kernels"] == []
    # No neighbours -> rate is undefined, not zero.
    assert metrics["same_kernel_neighbor_rate"] is None
    assert metrics["heldout_shape_leaked"] is False


def test_classify_failure_maps_messages_to_status() -> None:
    from helion_rag.loo_run import _classify_failure

    assert _classify_failure("RuntimeError: CUDA out of memory") == "oom"
    assert _classify_failure("AssertionError: accuracy check failed") == "accuracy_fail"
    assert _classify_failure("InvalidConfig: bad pid_type") == "compile_fail"
    assert _classify_failure("ValueError: something else") == "error"
