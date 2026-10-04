from __future__ import annotations

import dataclasses
import gc
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.utils._pytree import tree_flatten

from test.test_autotune_candidate_isolation import _provider

import helion
from helion import exc
from helion._testing import DEVICE
from helion.autotuner import benchmark_job
from helion.autotuner import benchmark_provider
from helion.autotuner import kernel_args
from helion.autotuner import precompile_future
from helion.runtime.cute import chained_startup

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

META_DEVICE = torch.device("meta")


@pytest.fixture(autouse=True)
def _clear_template_cache() -> Any:
    kernel_args.load_trusted_kernel_args.cache_clear()
    yield
    kernel_args.load_trusted_kernel_args.cache_clear()


def _guard(source: torch.Tensor, output: torch.Tensor) -> None:
    plan = SimpleNamespace(
        _helion_cute_wrapper_plans=[
            {
                "kind": "chained_startup_tma",
                "lhs_idx": 0,
                "out_idx": 1,
                "dtype": "bfloat16",
                "shape": tuple(source.shape),
                "strides": source.stride(),
            }
        ]
    )
    # CPU storage exercises the actual pointer, alignment, alias and span checks;
    # only device placement is modeled so these tests do not require a GPU.
    with patch.object(torch.Tensor, "device", torch.device("cuda", 0)):
        chained_startup.validate_arguments(plan, (source, output))


def _source(
    offset: int, *, external: bool = False, device: str | torch.device = "cpu"
) -> torch.Tensor:
    storage = torch.arange(2 * 128 * 128 + offset, device=device).to(torch.bfloat16)
    value = storage[offset:].view(2, 128, 128)
    return torch.from_dlpack(value) if external else value


def _same_contract(original: Sequence[object], private: Sequence[object]) -> None:
    left, left_spec = tree_flatten(original)
    right, right_spec = tree_flatten(private)
    assert left_spec == right_spec
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for a, b in zip(left, right, strict=True):
        if not isinstance(a, torch.Tensor):
            assert a == b
            continue
        assert isinstance(b, torch.Tensor)
        assert (
            a.shape,
            a.stride(),
            a.storage_offset(),
            a.dtype,
            a.device,
            a.is_conj(),
            a.is_neg(),
            a.requires_grad,
        ) == (
            b.shape,
            b.stride(),
            b.storage_offset(),
            b.dtype,
            b.device,
            b.is_conj(),
            b.is_neg(),
            b.requires_grad,
        )
        assert b.is_leaf
        assert a.untyped_storage().nbytes() == b.untyped_storage().nbytes()
        assert (
            a.untyped_storage().data_ptr() % 256 == b.untyped_storage().data_ptr() % 256
        )
        assert a.data_ptr() % 256 == b.data_ptr() % 256
        assert a.untyped_storage()._cdata != b.untyped_storage()._cdata
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        pairs.append((a, b))
    for a, b in pairs:
        for c, d in pairs:
            assert (a is c) == (b is d)
            assert (a.untyped_storage()._cdata == c.untyped_storage()._cdata) == (
                b.untyped_storage()._cdata == d.untyped_storage()._cdata
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("offset", [0, 1, 7, 8, 63, 64, 127, 128])
@pytest.mark.parametrize("external", [False, True])
def test_private_clone_alignment_boundaries(
    dtype: torch.dtype, offset: int, external: bool
) -> None:
    storage = torch.arange(512 + offset).to(dtype)
    value = storage[offset:].view(16, 32)
    if external:
        value = torch.from_dlpack(value)
    args = (value,)
    private = kernel_args._clone_args(args, None)
    _same_contract(args, private)
    cast("torch.Tensor", private[0]).zero_()
    assert torch.count_nonzero(value) > 0


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("offset", [0, 1, 8, 64, 128])
def test_real_startup_guard_is_unchanged_by_clone_and_transport(
    tmp_path: Path, external: bool, offset: int
) -> None:
    source = _source(offset, external=external)
    args = (source, torch.zeros_like(source))
    path = str(tmp_path / "args.pt")
    kernel_args.save_trusted_kernel_args(args, path)
    template = kernel_args.load_trusted_kernel_args(path)
    assert template is kernel_args.load_trusted_kernel_args(path)
    _same_contract(args, template)
    for values in (args, template, kernel_args._clone_args(template, None)):
        a, b = cast("tuple[torch.Tensor, torch.Tensor]", values)
        if source.data_ptr() % 16:
            with pytest.raises(exc.BackendUnsupported, match="alignment"):
                _guard(a, b)
        else:
            _guard(a, b)


@pytest.mark.parametrize(
    "kind", ["strided", "broadcast", "mixed", "conj", "neg", "empty"]
)
def test_storage_aliases_flags_and_partial_selection(kind: str) -> None:
    base = torch.arange(128, dtype=torch.float32)
    if kind == "conj":
        base = torch.complex(base, base + 1)
    base.requires_grad_()
    base = base * 2  # nonleaf inputs still produce independent leaf copies
    if kind == "strided":
        view = base.as_strided((3, 4), (9, 2), 1)
    elif kind == "broadcast":
        view = base[1:2].expand(3, 4)
    elif kind == "mixed":
        view = base.detach().view(torch.uint8)[3:31]
    elif kind == "conj":
        view = base[1:].conj()
    elif kind == "neg":
        view = torch._neg_view(base)[1:]
    else:
        view = base[1:1]
    untouched = torch.arange(3)
    args = (base, {"view": view}, view, untouched)
    private = kernel_args._clone_args(args, None, idx_to_clone=[1])
    assert private[3] is untouched
    _same_contract(args[:3], private[:3])
    before = base.detach().clone()
    with torch.no_grad():
        cast("torch.Tensor", private[0]).zero_()
    torch.testing.assert_close(base, before, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["strided", "conj", "neg", "empty"])
def test_serialized_aliases_flags_and_grad(tmp_path: Path, kind: str) -> None:
    base = torch.arange(128, dtype=torch.float32)
    if kind == "conj":
        base = torch.complex(base, base + 1)
    base = torch.from_dlpack(base[1:])
    base.requires_grad_()
    view = base[1::2]
    if kind == "conj":
        view = view.conj()
    elif kind == "neg":
        view = torch._neg_view(view)
    elif kind == "empty":
        view = base[1:1]
    args = (base, {"view": view}, view)
    path = str(tmp_path / "args.pt")
    kernel_args.save_trusted_kernel_args(args, path)
    restored = kernel_args.load_trusted_kernel_args(path)
    _same_contract(args, restored)
    _same_contract(args, kernel_args._clone_args(restored, None))


def test_default_contiguous_clone_call_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.arange(128)
    clone = torch.Tensor.clone
    calls: list[torch.Tensor] = []

    def tracked(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(tensor)
        return clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", tracked)
    result = kernel_args._clone_args((value, value), None)
    assert len(calls) == 1
    _same_contract((value, value), result)


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("offset", [1, 8])
def test_provider_rejects_original_misalignment(
    monkeypatch: pytest.MonkeyPatch, external: bool, offset: int
) -> None:
    source = _source(offset, external=external)
    seen: list[int] = []

    def candidate(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        seen.append(x.data_ptr() % 16)
        _guard(x, out)
        return x.clone()

    monkeypatch.setattr(benchmark_provider, "synchronize_device", lambda: None)
    timer = Mock(side_effect=lambda fn, **kwargs: (fn(), 1.0)[1])
    monkeypatch.setattr(benchmark_provider, "do_bench", timer)
    provider = _provider((source, torch.zeros_like(source)), lambda x, out: x.clone())
    cast("Any", provider.kernel).bench_compile_config.return_value = candidate
    assert provider.mutated_arg_indices == []
    result = provider._benchmark_function(helion.Config(), cast("Any", candidate))
    if offset == 1:
        assert math.isinf(result)
        assert seen == [2]
        timer.assert_not_called()
    else:
        assert result == 1.0
        assert seen == [0, 0, 0]
        timer.assert_called_once()


@pytest.mark.parametrize("entry", ["benchmark", "accuracy", "precompile"])
@pytest.mark.parametrize("offset", [1, 8])
def test_worker_entrypoints_use_restored_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, offset: int
) -> None:
    args = (
        _source(offset, external=True),
        torch.zeros((2, 128, 128), dtype=torch.bfloat16),
    )
    monkeypatch.setattr(benchmark_provider, "synchronize_device", lambda: None)
    provider = _provider(args, lambda x, out: x.clone())
    provider.settings.autotune_precompile = "spawn"
    # Exercise the sole production writer, not a test-only torch.save envelope.
    provider.setup()
    try:
        path = provider._precompile_args_path
        assert path is not None
        baseline = str(tmp_path / "baseline.pt")
        torch.save(args[0], baseline)
        seen: list[int] = []

        def candidate(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            seen.append(x.data_ptr() % 16)
            _guard(x, out)
            return x

        monkeypatch.setattr(
            benchmark_job, "_load_compiled_fn_for_worker", lambda spec: candidate
        )
        monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
        monkeypatch.setattr(benchmark_job, "synchronize_device", lambda: None)
        monkeypatch.setattr(
            benchmark_job, "do_bench", lambda fn, **kwargs: (fn(), 1.0)[1]
        )
        if entry == "benchmark":
            job = benchmark_job.BenchmarkJob(cast("Any", None), path)
            if offset == 1:
                with pytest.raises(exc.BackendUnsupported, match="alignment"):
                    job()
            else:
                assert job() == 1.0
        elif entry == "accuracy":
            check = benchmark_job.AccuracyCheckJob(
                cast("Any", None), path, baseline, 0, 0
            )
            if offset == 1:
                with pytest.raises(exc.BackendUnsupported, match="alignment"):
                    check()
            else:
                assert check().ok
        else:
            monkeypatch.setattr(
                precompile_future, "_load_compiled_fn", lambda spec: candidate
            )
            monkeypatch.setattr(
                precompile_future, "_unload_compiled_fn", lambda fn: None
            )
            monkeypatch.setattr(precompile_future, "synchronize_device", lambda: None)
            monkeypatch.setattr(
                precompile_future, "start_isolated_process_group", lambda: None
            )
            exit_process = Mock()
            monkeypatch.setattr(precompile_future.os, "_exit", exit_process)
            precompile_future._run_kernel_in_subprocess_spawn(
                cast("Any", None), path, str(tmp_path / "result"), "alignment"
            )
            exit_process.assert_called_once_with(int(offset == 1))
        assert seen == [2 if offset == 1 else 0]
        _same_contract(args, kernel_args.load_trusted_kernel_args(path))
    finally:
        provider.cleanup()


def test_worker_bad_then_good_keeps_cached_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(8, external=True)
    args = (source, source[:, :, ::2], source)
    path = str(tmp_path / "args.pt")
    kernel_args.save_trusted_kernel_args(args, path)
    pristine = kernel_args.load_trusted_kernel_args(path)
    seen: list[Sequence[object]] = []

    def candidate(*values: torch.Tensor) -> None:
        _same_contract(args, values)
        seen.append(values)
        if len(seen) == 1:
            values[0].fill_(99)

    monkeypatch.setattr(
        benchmark_job, "_load_compiled_fn_for_worker", lambda spec: candidate
    )
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(benchmark_job, "do_bench", lambda fn, **kwargs: (fn(), 1.0)[1])
    for _ in range(2):
        assert benchmark_job.BenchmarkJob(cast("Any", None), path)() == 1.0
        assert pristine is kernel_args.load_trusted_kernel_args(path)
        _same_contract(args, pristine)
    assert (
        cast("torch.Tensor", seen[0][0]).data_ptr()
        != cast("torch.Tensor", seen[1][0]).data_ptr()
    )


@pytest.mark.parametrize(
    "field", ["version", "bool_version", "residue", "bool_residue", "count", "layout"]
)
def test_malformed_envelope_fails(tmp_path: Path, field: str) -> None:
    path = str(tmp_path / "args.pt")
    args = (_source(1, external=True),)
    layouts, residues = kernel_args._argument_layouts(args)
    payload = kernel_args._TrustedKernelArgs(1, args, layouts, residues)
    if field == "version":
        payload = dataclasses.replace(payload, version=2)
    elif field == "bool_version":
        payload = dataclasses.replace(payload, version=True)
    elif field == "residue":
        payload = dataclasses.replace(payload, residues=(256,))
    elif field == "bool_residue":
        payload = dataclasses.replace(payload, residues=(True,))
    elif field == "count":
        payload = dataclasses.replace(payload, residues=())
    else:
        payload = dataclasses.replace(
            payload, layouts=(dataclasses.replace(layouts[0], offset=9),)
        )
    torch.save(payload, path)
    with pytest.raises(ValueError, match="trusted kernel argument"):
        kernel_args.load_trusted_kernel_args(path)
    assert kernel_args.load_trusted_kernel_args.cache_info().currsize == 0


def test_legacy_python_objects_and_mixed_dtype_boundary(tmp_path: Path) -> None:
    path = str(tmp_path / "legacy.pt")
    args = (SimpleNamespace(value=3), torch.arange(8))
    torch.save(args, path)
    loaded = kernel_args.load_trusted_kernel_args(path)
    assert loaded[0] == args[0]
    current = str(tmp_path / "current.pt")
    kernel_args.save_trusted_kernel_args(args, current)
    _same_contract(args, kernel_args.load_trusted_kernel_args(current))
    value = torch.arange(16, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same data as different types"):
        kernel_args.save_trusted_kernel_args(
            (value, value.view(torch.uint8)), str(tmp_path / "mixed.pt")
        )


def test_memory_accounting_uses_full_storage_spans() -> None:
    storage = torch.arange(1024)
    args = (storage[1:2], storage[20:21], storage[1:2])
    assert (
        kernel_args._argument_storage_bytes(args) == storage.untyped_storage().nbytes()
    )


@pytest.mark.parametrize("external", [False, True])
def test_standard_parameter_offset_is_preserved(tmp_path: Path, external: bool) -> None:
    parameter = torch.nn.Parameter(_source(1, external=external))
    args = (parameter, parameter, parameter[:, :, ::2])
    _same_contract(args, kernel_args._clone_args(args, None))
    path = str(tmp_path / "parameter.pt")
    kernel_args.save_trusted_kernel_args(args, path)
    _same_contract(args, kernel_args.load_trusted_kernel_args(path))


@pytest.mark.parametrize("kind", ["sparse", "quantized", "meta"])
def test_special_tensor_clone_paths_are_retained(kind: str) -> None:
    if kind == "sparse":
        tensor = torch.eye(4).to_sparse()
    elif kind == "quantized":
        tensor = torch.quantize_per_tensor(torch.arange(8.0), 0.1, 0, torch.qint8)
    else:
        tensor = torch.empty((3, 4), device=META_DEVICE)
    cloned = cast("torch.Tensor", kernel_args._clone_args((tensor,), None)[0])
    assert cloned.shape == tensor.shape
    assert cloned.layout == tensor.layout
    assert cloned.dtype == tensor.dtype
    assert cloned.device == tensor.device
    if kind == "sparse":
        torch.testing.assert_close(cloned.to_dense(), tensor.to_dense())
    elif kind == "quantized":
        torch.testing.assert_close(cloned.dequantize(), tensor.dequantize())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("offset", [1, 8])
@pytest.mark.parametrize("external", [False, True])
def test_cuda_private_storage_transfer_lifetime(
    tmp_path: Path, offset: int, external: bool
) -> None:
    source = _source(offset, external=external, device=DEVICE)
    args = (source, torch.zeros_like(source))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        private = kernel_args._clone_args(args, None)
        gc.collect()  # byte-window owners must survive through tensor storage
        result = cast("torch.Tensor", private[0]) + 1
    stream.synchronize()
    _same_contract(args, private)
    torch.testing.assert_close(result, source + 1, atol=0, rtol=0)
    path = str(tmp_path / "cuda.pt")
    kernel_args.save_trusted_kernel_args(args, path)
    restored = kernel_args.load_trusted_kernel_args(path)
    _same_contract(args, restored)
    for values in (private, restored):
        a, b = cast("tuple[torch.Tensor, torch.Tensor]", values)
        if offset == 1:
            with pytest.raises(exc.BackendUnsupported, match="alignment"):
                _guard(a, b)
        else:
            _guard(a, b)
    torch.cuda.synchronize()
