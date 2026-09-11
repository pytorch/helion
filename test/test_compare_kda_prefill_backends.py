from __future__ import annotations

import inspect
import sys
from types import ModuleType
from types import SimpleNamespace

from benchmarks.cute import compare_kda_prefill_backends as prefill
import pytest
import torch


def _inputs() -> prefill.KDAInputs:
    value = torch.ones(1)
    return prefill.KDAInputs(
        q=value,
        k=value,
        v=value,
        raw_gate=value,
        raw_beta=value,
        a_log=value,
        dt_bias=value,
        initial_state=value.clone(),
        cu_seqlens=None,
    )


def test_clone_mutable_shares_inputs_and_clones_only_state() -> None:
    inputs = _inputs()
    clone = inputs.clone_mutable()

    assert clone.q is inputs.q
    assert clone.k is inputs.k
    assert clone.v is inputs.v
    assert clone.raw_gate is inputs.raw_gate
    assert clone.raw_beta is inputs.raw_beta
    assert clone.initial_state is not inputs.initial_state
    torch.testing.assert_close(clone.initial_state, inputs.initial_state)


def test_surface_contains_only_reportable_cases_and_providers() -> None:
    assert tuple(prefill.CASES) == ("h12_fixed_512", "h12_packed_mixed")
    assert tuple(prefill.CASES) == prefill.OFFICIAL_CASES
    assert prefill.IMPLEMENTATIONS == (
        "helion-cute",
        "flashinfer-cake",
        "flashinfer-cute",
    )
    assert prefill.CASES["h12_fixed_512"].seq_lens == (512,)
    assert prefill.CASES["h12_packed_mixed"].seq_lens == (
        1300,
        547,
        2048,
        963,
        271,
        3063,
    )
    assert all(
        case.expected_cake_variant == "bt16_prepare_chain_m64"
        for case in prefill.CASES.values()
    )
    assert {
        name: config.as_dict() for name, config in prefill.PINNED_HELION_CONFIGS.items()
    } == {
        "h12_fixed_512": {
            "staged_topology": "serial",
            "critical_prepare_schedule": "split_alias_cpc1",
            "recurrence_dv_partitions": 4,
            "recurrence_register_cap": None,
        },
        "h12_packed_mixed": {
            "staged_topology": "origin_aux",
            "critical_prepare_schedule": "split_alias_cpc2",
            "auxiliary_prepare_schedule": "split_alias_cpc2",
            "recurrence_dv_partitions": 2,
            "recurrence_register_cap": None,
        },
    }


def test_executable_surface_is_self_contained_and_fixed(tmp_path) -> None:
    parser = prefill._build_parser()
    args = parser.parse_args(
        [
            "--paired-cuda-graph",
            "--flashinfer-root",
            str(tmp_path / "flashinfer"),
        ]
    )

    prefill._validate_args(parser, args)
    assert args.cases == ",".join(prefill.OFFICIAL_CASES)
    assert args.impls == ",".join(prefill.IMPLEMENTATIONS)
    assert "run_kda_prefill_cold_autotune" not in inspect.getsource(prefill.main)


def test_flashinfer_provenance_covers_cake_and_cute_sources(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_git_files(root, pathspecs):
        captured["pathspecs"] = pathspecs
        return {
            "flashinfer/kda_kernels/kda_chunked_bt16.py": (
                root / "flashinfer/kda_kernels/kda_chunked_bt16.py"
            )
        }

    monkeypatch.setattr(prefill, "_git_files", fake_git_files)
    sources = prefill._flashinfer_sources(tmp_path)

    assert captured["pathspecs"] == (
        "csrc/kda",
        "flashinfer/cute_dsl",
        "flashinfer/kda_kernels",
    )
    assert "flashinfer/kda_prefill.py" in sources
    assert "flashinfer/kda_prefill_cute.py" in sources
    assert "flashinfer/kda_kernels/kda_chunked_bt16.py" in sources


@pytest.mark.parametrize(
    "prefill_case", prefill.CASES.values(), ids=lambda case: case.name
)
def test_semantic_metadata_records_reportable_contract(prefill_case) -> None:
    metadata = prefill._semantic_metadata(prefill_case)

    assert metadata["qk_l2norm_epsilon"] == 1.0e-24
    assert metadata["gate_mode"] == "lower-bound-sigmoid"
    assert metadata["canonical_state_dtype"] == "bfloat16"
    assert metadata["timing"] == "one-call-captured-graph-cuda-events-with-cold-l2"
    assert metadata["mutation_isolation"] == (
        "private-state-and-output-per-captured-graph"
    )


def test_loader_rejects_preloaded_namespace_from_another_checkout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "test_kda_prefill_namespace"
    existing = ModuleType(name)
    existing.__path__ = [str(tmp_path / "wrong")]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, existing)

    with pytest.raises(ImportError, match="outside the requested checkout"):
        prefill._install_namespace(name, tmp_path / "expected")


def test_loader_rejects_preloaded_module_from_another_checkout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "test_kda_prefill_module"
    existing = ModuleType(name)
    existing.__file__ = str(tmp_path / "wrong.py")
    monkeypatch.setitem(sys.modules, name, existing)

    with pytest.raises(ImportError, match="outside the requested checkout"):
        prefill._load_module(name, tmp_path / "expected.py")


def test_safe_gate_route_accepts_only_combined_or_prepare_chain() -> None:
    assert prefill._resolve_safe_gate_modules(
        [{"variant": "bt16_prepare_chain_m64_s8", "target": "sm100"}]
    ) == (
        "bt16_prepare_chain_m64",
        "sm100",
        ["bt16_prepare_chain_m64_s8"],
    )
    assert prefill._resolve_safe_gate_modules(
        [
            {"variant": "bt16_prepare_beta_tma", "target": "sm100"},
            {"variant": "bt16_chain_m64_s9", "target": "sm100"},
        ]
    ) == (
        "bt16_prepare_chain_m64",
        "sm100",
        ["bt16_prepare_beta_tma", "bt16_chain_m64_s9"],
    )
    with pytest.raises(AssertionError, match="ordered BT16 prepare/chain"):
        prefill._resolve_safe_gate_modules(
            [{"variant": "m128_h12_long", "target": "sm100"}]
        )


def test_flashinfer_invocation_forwards_the_canonical_contract() -> None:
    calls = []

    class FlashInfer:
        @staticmethod
        def recurrent_kda(**kwargs):
            calls.append(kwargs)
            return kwargs["output"], None

    inputs = _inputs()
    output = torch.empty_like(inputs.v)
    result_output, result_state = prefill._invoke_flashinfer(
        FlashInfer(), prefill.CASES["h12_fixed_512"], "cake"
    )(inputs, output)

    assert result_output is output
    assert result_state is inputs.initial_state
    assert calls == [
        {
            "q": inputs.q,
            "k": inputs.k,
            "v": inputs.v,
            "g": inputs.raw_gate,
            "beta": inputs.raw_beta,
            "A_log": inputs.a_log,
            "dt_bias": inputs.dt_bias,
            "scale": prefill.HEAD_DIM**-0.5,
            "initial_state": inputs.initial_state,
            "output_final_state": False,
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "lower_bound": prefill.SAFE_LOWER_BOUND,
            "cu_seqlens": None,
            "ssm_state_indices": None,
            "output": output,
            "beta_is_logit": True,
            "state_checkpoints": None,
            "checkpoint_cu_starts": None,
            "checkpoint_every_n_tokens": 0,
            "backend": "cake",
        }
    ]


def test_route_recorder_validates_cake_module_and_target() -> None:
    prefill_module = SimpleNamespace(
        _get_flash_kda_prefill_module=lambda variant, target: object(),
        _select_flash_kda_prefill_target=lambda _device: "sm100",
    )
    cute_module = SimpleNamespace(_run_cute_dsl_kda_prefill=lambda **_kwargs: None)
    module = SimpleNamespace(
        _kda_prefill=prefill_module,
        _kda_prefill_cute=cute_module,
    )
    inputs = _inputs()
    metadata = {}

    def invoke(call_inputs, output):
        module._kda_prefill._get_flash_kda_prefill_module(
            "bt16_prepare_chain_m64_s8", "sm100"
        )
        return output, call_inputs.initial_state

    wrapped = prefill._record_flashinfer_route(
        module,
        invoke,
        metadata,
        case=prefill.CASES["h12_fixed_512"],
        expected_route="cake-safe-gate",
    )
    wrapped(inputs, torch.empty_like(inputs.v))

    assert metadata["resolved_route"] == "cake-safe-gate"
    assert metadata["cake_logical_variant"] == "bt16_prepare_chain_m64"
    assert metadata["cake_target"] == "sm100"
    assert metadata["cake_physical_variants"] == ["bt16_prepare_chain_m64_s8"]
