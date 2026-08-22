from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

from benchmarks.cute import grouped_gemm_deepgemm_support as support
from pretuned_kernels.grouped_gemm_deepgemm import _deepgemm_public_api
import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path


# CPU tensors and mocked DeepGEMM modules isolate these host-side contracts from CUDA.
def test_repack_case_alignment_preserves_logical_values() -> None:
    torch.manual_seed(0)
    actual_ms = (2, 3)
    a, b, layout, reference, worklist = support.make_case(
        2,
        3,
        4,
        actual_ms,
        torch.device("cpu"),
        4,
    )

    repacked_a, repacked_b, repacked_layout, repacked_reference, repacked_worklist = (
        support.repack_case_alignment(
            a,
            b,
            reference,
            worklist,
            actual_ms,
            3,
        )
    )

    assert repacked_b is b
    assert repacked_worklist.tolist() == [[0, 0, 2, 3], [1, 3, 3, 3]]
    assert repacked_layout.tolist() == [0, 0, -1, 1, 1, 1]
    for source, target, actual_m in ((0, 0, 2), (4, 3, 3)):
        torch.testing.assert_close(
            repacked_a[target : target + actual_m],
            a[source : source + actual_m],
        )
        torch.testing.assert_close(
            repacked_reference[target : target + actual_m],
            reference[source : source + actual_m],
        )
    assert torch.count_nonzero(repacked_a[2]) == 0
    assert layout.tolist() == [0, 0, -1, -1, 1, 1, 1, -1]


@pytest.mark.parametrize(
    ("output_values", "max_diff", "require_zero_padding", "expected"),
    (
        ([1.0, 13.0, 2.0], 0.0, False, True),
        ([1.0, 13.0, 2.0], 0.0, True, False),
        ([1.0, 13.0, 3.0], 1e-5, False, False),
    ),
)
def test_correctness_padding_contract(
    output_values: list[float],
    max_diff: float,
    require_zero_padding: bool,
    expected: bool,
) -> None:
    reference = torch.tensor([[1.0], [0.0], [2.0]])
    layout = torch.tensor([0, -1, 1], dtype=torch.int32)
    output = torch.tensor(output_values).unsqueeze(1)

    assert (
        support.correctness(
            output,
            reference,
            layout,
            max_diff=max_diff,
            require_zero_padding=require_zero_padding,
        )["ok"]
        is expected
    )


def test_import_deepgemm_records_public_module_and_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "DeepGEMM"
    module_path = root / "deep_gemm" / "__init__.py"
    extension = root / "deep_gemm" / "_C.cpython-312-x86_64-linux-gnu.so"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("")
    extension.write_bytes(b"extension")
    for dependency in ("cutlass", "fmt"):
        (root / "third-party" / dependency).mkdir(parents=True)

    heads: list[tuple[Path, str, str]] = []
    monkeypatch.setattr(
        support,
        "_clean_checkout",
        lambda path, expected, label: heads.append((path, expected, label)) or expected,
    )
    monkeypatch.setattr(
        support,
        "_native_extension",
        lambda _root: (
            extension,
            {"path": extension.relative_to(root).as_posix(), "sha256": "abc"},
        ),
    )
    alignments: list[int] = []
    runtime: dict[str, object] = {}
    module = SimpleNamespace(
        __file__=str(module_path),
        __version__=support.DEEPGEMM_VERSION,
        _C=SimpleNamespace(__file__=str(extension)),
        set_mk_alignment_for_contiguous_layout=alignments.append,
        get_mk_alignment_for_contiguous_layout=lambda: alignments[-1],
        get_theoretical_mk_alignment_for_contiguous_layout=lambda: support.M_ALIGNMENT,
        set_num_sms=lambda value: runtime.__setitem__("num_sms", value),
        get_num_sms=lambda: 148 if runtime["num_sms"] == 0 else runtime["num_sms"],
        set_tc_util=lambda value: runtime.__setitem__("tc_util", value),
        get_tc_util=lambda: runtime["tc_util"],
        set_pdl=lambda value: runtime.__setitem__("pdl", value),
        get_pdl=lambda: runtime["pdl"],
        set_ignore_compile_dims=lambda value: runtime.__setitem__(
            "ignore_compile_dims", value
        ),
        set_block_size_multiple_of=lambda value: runtime.__setitem__(
            "block_size_multiple_of", value
        ),
    )
    monkeypatch.setattr(support.importlib, "import_module", lambda _name: module)

    imported, provenance = support.import_deepgemm(root, support.M_ALIGNMENT)

    assert imported is module
    assert alignments == [support.M_ALIGNMENT]
    assert provenance["git_head"] == support.DEEPGEMM_COMMIT
    assert provenance["m_alignment"] == support.M_ALIGNMENT
    assert provenance["native_extension"]["sha256"] == "abc"
    assert provenance["runtime_controls"] == {
        "num_sms": 148,
        "tc_util": 100,
        "pdl": False,
        "ignore_compile_dims": False,
        "block_size_multiple_of": 1,
    }
    assert [item[2] for item in heads] == [
        "DeepGEMM",
        "DeepGEMM CUTLASS",
        "DeepGEMM fmt",
    ]


def test_import_deepgemm_rejects_module_outside_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "DeepGEMM"
    root.mkdir()
    extension = root / "_C.so"
    extension.write_bytes(b"extension")
    monkeypatch.setattr(support, "_clean_checkout", lambda *_args: "head")
    monkeypatch.setattr(
        support,
        "_native_extension",
        lambda _root: (extension, {}),
    )
    module = SimpleNamespace(
        __file__=str(tmp_path / "foreign" / "__init__.py"),
        __version__=support.DEEPGEMM_VERSION,
        _C=SimpleNamespace(__file__=str(extension)),
    )
    monkeypatch.setattr(support.importlib, "import_module", lambda _name: module)

    with pytest.raises(RuntimeError, match="outside the validated checkout"):
        support.import_deepgemm(root, support.M_ALIGNMENT)


def test_import_deepgemm_rejects_control_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DG_JIT_DEBUG", "1")

    with pytest.raises(RuntimeError, match="DG_JIT_DEBUG"):
        support.import_deepgemm(tmp_path, support.M_ALIGNMENT)


def test_effective_reviewed_config_checks_requested_and_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = {"block_sizes": [256, 128, 64], "num_warps": 8}
    profile = SimpleNamespace(config_name="reviewed")
    monkeypatch.setattr(
        _deepgemm_public_api._REVIEWED,
        "reviewed_config_values",
        lambda _name: requested,
    )

    class ConfigSpec:
        @staticmethod
        def normalized_config(config: Any) -> SimpleNamespace:
            values = config.config if hasattr(config, "config") else config
            return SimpleNamespace(config={**values, "normalized": True})

    bound = SimpleNamespace(
        _config=SimpleNamespace(config=requested),
        config_spec=ConfigSpec(),
    )

    assert _deepgemm_public_api._effective_reviewed_config(bound, profile) == {
        "requested": requested,
        "effective": {**requested, "normalized": True},
    }
    bound._config = SimpleNamespace(config={"num_warps": 4})
    with pytest.raises(RuntimeError, match="exact reviewed config"):
        _deepgemm_public_api._effective_reviewed_config(bound, profile)
