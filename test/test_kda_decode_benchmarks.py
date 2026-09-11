from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from benchmarks.cute import compare_kda_decode_backends as packed
from benchmarks.cute import compare_kda_recurrent_backends as recurrent
from benchmarks.cute import kda_benchmark_utils as benchmark_utils
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_imports_resolve_inside_checkout() -> None:
    for module in (packed, recurrent, benchmark_utils):
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(REPO_ROOT)
    assert packed.helion.__file__ is not None
    assert Path(packed.helion.__file__).resolve().is_relative_to(REPO_ROOT)


@pytest.mark.parametrize(
    "script",
    (
        "compare_kda_decode_backends.py",
        "compare_kda_recurrent_backends.py",
    ),
)
def test_benchmark_direct_entrypoint_imports_clean_checkout(
    script: str, tmp_path: Path
) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "cute" / script), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_sglang_policy_constant_is_read_from_pinned_source(tmp_path: Path) -> None:
    source = tmp_path / "__init__.py"
    source.write_text("KDA_SMALL_VALUE_HEAD_THRESHOLD = 17\n")
    assert (
        packed._read_source_int_constant(source, "KDA_SMALL_VALUE_HEAD_THRESHOLD") == 17
    )
    source.write_text("KDA_SMALL_VALUE_HEAD_THRESHOLD = compute_threshold()\n")
    with pytest.raises(ValueError, match="expected one literal integer"):
        packed._read_source_int_constant(source, "KDA_SMALL_VALUE_HEAD_THRESHOLD")


def test_paired_graph_order_is_balanced() -> None:
    assert benchmark_utils.paired_graph_order(0, "baseline", "candidate") == (
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    )
    assert benchmark_utils.paired_graph_order(1, "baseline", "candidate") == (
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    )


def test_source_provenance_includes_untracked_python(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    package = tmp_path / "package"
    package.mkdir()
    tracked = package / "tracked.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(("git", "add", "package/tracked.py"), cwd=tmp_path, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=KDA Test",
            "-c",
            "user.email=kda@example.invalid",
            "commit",
            "-qm",
            "initial",
        ),
        cwd=tmp_path,
        check=True,
    )
    clean = benchmark_utils.source_provenance(
        tmp_path, {"tracked": tracked}, pathspecs=("package",)
    )
    (package / "untracked.py").write_text("VALUE = 2\n")
    dirty = benchmark_utils.source_provenance(
        tmp_path, {"tracked": tracked}, pathspecs=("package",)
    )

    assert clean["dirty"] is False
    assert dirty["dirty"] is True
    assert dirty["status_includes_untracked"] is True
    assert dirty["untracked_python_sha256"] != clean["untracked_python_sha256"]


def test_source_provenance_scopes_untracked_python(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    package = tmp_path / "package"
    package.mkdir()
    tracked = package / "tracked.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(("git", "add", "package/tracked.py"), cwd=tmp_path, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=KDA Test",
            "-c",
            "user.email=kda@example.invalid",
            "commit",
            "-qm",
            "initial",
        ),
        cwd=tmp_path,
        check=True,
    )
    clean = benchmark_utils.source_provenance(
        tmp_path, {"tracked": tracked}, pathspecs=("package",)
    )
    (tmp_path / "outside.py").write_text("VALUE = 2\n")
    unchanged = benchmark_utils.source_provenance(
        tmp_path, {"tracked": tracked}, pathspecs=("package",)
    )

    assert unchanged["dirty"] is False
    assert unchanged["untracked_python_sha256"] == clean["untracked_python_sha256"]


def _fake_bound(source: str, compiled_hash: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        _run=object(),
        env=SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda compiled: compiled_hash
            )
        ),
        to_triton_code=lambda config: source,
    )


def test_generated_wrapper_verification_records_exact_source_identity() -> None:
    source = "expected_plan_abi_version = 7\n_bf16x2_12_abi_version = 1\n"
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = benchmark_utils.verify_helion_generated_wrapper(
        _fake_bound(source, source_hash),
        object(),
        expected_plan_kind="expected-plan",
        expected_source_markers=("expected_plan_abi_version = 7",),
        expected_source_patterns=(r"_bf16x2_[0-9]+_abi_version = 1",),
    )

    assert identity["plan_kind"] == "expected-plan"
    assert identity["source_sha256"] == source_hash
    assert identity["source_marker_counts"] == {"expected_plan_abi_version = 7": 1}
    assert identity["source_pattern_counts"] == {r"_bf16x2_[0-9]+_abi_version = 1": 1}
    assert identity["compiled_source_identity_verified"] is True


def test_generated_wrapper_verification_accepts_direct_compiled_callable() -> None:
    source = "expected_plan_abi_version = 7\n"
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    bound = _fake_bound(source, source_hash)
    bound._run = None

    identity = benchmark_utils.verify_helion_generated_wrapper(
        bound,
        object(),
        expected_plan_kind="expected-plan",
        expected_source_markers=("expected_plan_abi_version = 7",),
        compiled_fn=object(),
    )

    assert identity["compiled_source_identity_verified"] is True


@pytest.mark.parametrize(
    ("compiled_hash", "marker", "match"),
    (
        ("wrong-hash", "expected_plan_abi_version = 7", "identity mismatch"),
        (None, "expected_plan_abi_version = 7", "identity mismatch"),
        ("source-hash", "missing_plan_abi_version = 1", "markers"),
    ),
)
def test_generated_wrapper_verification_rejects_unproven_plan(
    compiled_hash: str | None, marker: str, match: str
) -> None:
    source = "expected_plan_abi_version = 7\n"
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    resolved_hash = source_hash if compiled_hash == "source-hash" else compiled_hash
    with pytest.raises(RuntimeError, match=match):
        benchmark_utils.verify_helion_generated_wrapper(
            _fake_bound(source, resolved_hash),
            object(),
            expected_plan_kind="expected-plan",
            expected_source_markers=(marker,),
        )


@pytest.mark.parametrize(
    ("batch_size", "config", "expected"),
    (
        (
            1,
            {},
            ("single-token-rank1", ("rank1_helper_abi_version = 4",), ()),
        ),
        (
            256,
            {
                "block_sizes": [128],
                "cute_async_load_stages": 5,
                "cute_async_load_group_rows": 2,
                "cute_async_store_policy": "l2_evict_last",
                "cute_bf16x2_recurrence": True,
                "cute_vector_widths": [8, 1, 1, 1],
                "num_threads": [8, 16],
            },
            (
                "generic-async-state-ring-bf16x2",
                (
                    (
                        "from helion._compiler.cute.l2_policy import "
                        "store_u32x4_l2_evict_last as "
                        "_cute_store_u32x4_l2_evict_last"
                    ),
                    "cute.arch.cp_async_wait_group(0)",
                ),
                (
                    (
                        r"_async_state_[0-9]+_smem = "
                        r"cutlass\.Array\(cutlass\.Uint16, 10240, "
                        r"space=cutlass\.AddressSpace\.smem, alignment=128\)"
                    ),
                    (
                        r"_async_state_[0-9]+_scalar_pairs = "
                        r"cute\.make_rmem_tensor\(8, cutlass\.Uint32\)"
                    ),
                    r"_bf16x2_[0-9]+_abi_version = 1",
                    r"_cute_store_u32x4_l2_evict_last\(.*\)",
                ),
            ),
        ),
        (
            256,
            {
                "cute_async_load_stages": 0,
                "cute_async_store_policy": "default",
                "cute_bf16x2_recurrence": False,
            },
            ("generic-decode", ("@cute.kernel",), ()),
        ),
        (
            256,
            {
                "block_sizes": [128],
                "cute_async_load_stages": 4,
                "cute_async_load_group_rows": 2,
                "cute_async_store_policy": "l2_evict_last",
                "cute_bf16x2_recurrence": True,
                "cute_vector_widths": [8, 1, 1, 1],
                "num_threads": [8, 16],
            },
            ("generic-decode", ("@cute.kernel",), ()),
        ),
    ),
)
def test_packed_codegen_expectations_cover_claimed_rows(
    batch_size: int,
    config: dict[str, object],
    expected: tuple[str, tuple[str, ...], tuple[str, ...]],
) -> None:
    assert packed._helion_codegen_expectation(batch_size, config) == expected


@pytest.mark.parametrize(
    ("case_name", "expected"),
    (
        (
            "t1-unbounded-b15-h32",
            (
                "split-single-token-rank1",
                (
                    "split_t1_codegen_abi_version = 1",
                    "split_t1_rank1_helper_abi_version = 4",
                ),
            ),
        ),
        (
            "t3-lower-bound-n8-h16",
            (
                "fixed-token-rank1",
                (
                    "fixed_rank1_codegen_abi_version = 3",
                    "fixed_rank1_rank1_helper_abi_version = 4",
                ),
            ),
        ),
    ),
)
def test_recurrent_codegen_expectations_cover_claimed_rows(
    case_name: str, expected: tuple[str, tuple[str, ...]]
) -> None:
    assert recurrent._helion_codegen_expectation(recurrent.CASES[case_name]) == expected


def test_cli_scope_is_only_canonical_paired_comparisons() -> None:
    assert packed.DEFAULT_BATCH_SIZES == (1, 256)
    assert packed.IMPLEMENTATIONS == ("flashinfer-cake", "helion-cute")
    assert tuple(recurrent.CASES) == recurrent.DEFAULT_CASES
    assert recurrent.IMPLEMENTATIONS == ("flashinfer-cute", "helion-cute")


@pytest.mark.parametrize(
    "value",
    ('{"num_warps": 4, "num_warps": 8}', '{"num_warps": NaN}', "[]"),
)
def test_config_parser_rejects_ambiguous_json(value: str) -> None:
    with pytest.raises(ValueError):
        benchmark_utils.canonical_config_json(value, "test config")


def test_recurrent_config_map_uses_strict_json(tmp_path: Path) -> None:
    config_map = tmp_path / "configs.json"
    config_map.write_text('{"t1-unbounded-b15-h32": {"num_warps": 4, "num_warps": 8}}')
    with pytest.raises(ValueError, match="duplicate key"):
        recurrent._load_helion_config_map(config_map)
