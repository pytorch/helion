from __future__ import annotations

from contextlib import contextmanager
import inspect
from types import FunctionType
from typing import TYPE_CHECKING
from unittest.mock import patch

from cutlass.base_dsl.dsl import BaseDSL
import pytest
import torch

from .test_cute_chunk_recurrence import _code
from helion._compiler.cute import chunk_recurrence
from helion._compiler.cute import chunk_recurrence_sm100 as original
from helion._compiler.cute import prepared_tcgen_edge as shared
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.prepared_tcgen_binding import PreparedProjectionHost

if TYPE_CHECKING:
    from collections.abc import Iterator

_FUNCTIONS = (
    (original, "host_chain_dv2"),
    (original, "kernel_chain_dv2"),
    (original, "_issue_prepared_state_k"),
    (original, "tcgen05_chain_stage_vmx_input_tmem"),
    (shared, "_prepare_descriptor_issue"),
    (shared, "_issue_atom"),
    (shared, "_read_companion"),
    (shared, "execute_prepared_wait"),
    (shared, "execute_prepared_issue"),
    (shared, "execute_prepared_read"),
    (shared, "execute_prepared_publication"),
)


@pytest.fixture(scope="module")
def host() -> PreparedProjectionHost:
    hosts = []
    planner = chunk_recurrence._plan_chunk_recurrence

    def capture(*args, **kwargs):
        plan = planner(*args, prepared_edge=True, **kwargs)
        assert plan is not None
        hosts.append(PreparedProjectionHost(plan))
        return plan

    with patch.object(chunk_recurrence, "_plan_chunk_recurrence", capture):
        _code(fp32_state=True, dv_partitions=2, pipeline="wide")
    assert len(hosts) == 1
    hosts[0].check()
    return hosts[0]


@contextmanager
def _original_function(function: FunctionType) -> Iterator[FunctionType]:
    inner = vars(function)["__wrapped__"]
    assert isinstance(inner, FunctionType)
    code, fields = inner.__code__, dict(vars(inner))
    try:
        yield inner
    finally:
        inner.__code__ = code
        vars(inner).clear()
        vars(inner).update(fields)


def _preprocess(inner: FunctionType) -> None:
    # Exercise the installed real lazy transition without compiling/invoking
    # a JIT or device function. Compiler and CUDA entry points remain forbidden.
    with (
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
    ):
        BaseDSL._preprocess_and_replace_code(inner)


@pytest.mark.parametrize(("module", "name"), _FUNCTIONS)
def test_actual_preprocessor_keeps_original_decorated_source(host, module, name):
    function = vars(module)[name]
    context = host._context
    before_cuda = torch.cuda.is_initialized()
    with _original_function(function) as inner:
        original_code = inner.__code__
        source = inspect.getsource(function)
        host.check()
        _preprocess(inner)
        assert vars(inner)["_original_code"] is original_code
        assert inner.__code__ is not original_code
        assert inspect.getsource(original_code) == source
        assert source.split("\n", 1)[1] == inspect.getsource(function)
        host.check()
        assert host._context is context
    host.check()
    assert torch.cuda.is_initialized() == before_cuda


@pytest.mark.parametrize("preprocessed", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "original_code",
        "wrapper_code",
        "wrapped",
        "closure",
        "kind",
        "helper",
        "preprocess",
        "location",
    ],
)
def test_original_dependency_mutations_reject(host, preprocessed, mutation):
    function = original.host_chain_dv2
    with _original_function(function) as inner:
        if preprocessed:
            _preprocess(inner)
        host.check()
        with patch.dict(vars(inner)), patch.dict(vars(function)):
            if mutation == "original_code":
                code = vars(inner).get("_original_code", inner.__code__)
                # A structurally equal clone must still invalidate identity.
                vars(inner)["_original_code"] = code.replace()
            elif mutation == "wrapper_code":
                old_code = function.__code__
                function.__code__ = old_code.replace()
                try:
                    with pytest.raises(_UnsupportedChain):
                        host.check()
                finally:
                    function.__code__ = old_code
                return
            elif mutation == "wrapped":
                vars(function)["__wrapped__"] = lambda: None
            elif mutation in ("closure", "kind"):
                at = 1 if mutation == "closure" else 0
                assert function.__closure__ is not None
                cell = function.__closure__[at]
                old = cell.cell_contents
                cell.cell_contents = (lambda: None) if at else "_kernel_helper"
                try:
                    with pytest.raises(_UnsupportedChain):
                        host.check()
                finally:
                    cell.cell_contents = old
                return
            elif mutation == "helper":
                with (
                    patch.object(shared, "execute_prepared_read", lambda: None),
                    pytest.raises(_UnsupportedChain),
                ):
                    host.check()
                return
            elif mutation == "preprocess":
                vars(inner)["_preprocess_enabled"] = False
            else:
                vars(inner)["_decorator_location"] = "foreign location"
            with pytest.raises(_UnsupportedChain):
                host.check()
        host.check()
    host.check()


def test_original_anchor_deletion_after_real_transition_rejects(host):
    with _original_function(original.host_chain_dv2) as inner:
        _preprocess(inner)
        host.check()
        vars(inner).pop("_original_code")
        with pytest.raises(_UnsupportedChain):
            host.check()
    host.check()


@pytest.mark.parametrize("preprocessed", [False, True])
@pytest.mark.parametrize("mutation", ["body", "decorator"])
def test_full_original_source_mutations_reject(host, preprocessed, mutation):
    with _original_function(original.host_chain_dv2) as inner:
        if preprocessed:
            _preprocess(inner)
        code = vars(inner).get("_original_code", inner.__code__)
        getsource = inspect.getsource
        text = getsource(code)
        changed = (
            text.replace("@cute.jit", "@cute.kernel", 1)
            if mutation == "decorator"
            else text + "    raise RuntimeError('foreign body')\n"
        )
        assert changed != text

        def source(value):
            return changed if value is code else getsource(value)

        with (
            patch.object(inspect, "getsource", source),
            pytest.raises(_UnsupportedChain),
        ):
            host.check()
    host.check()


def test_explicit_original_source_not_transformed_bytecode_attestation(host):
    # The dependency witness covers original code, not an integrity monitor of
    # CuTe's transformed code. Keep this boundary explicit instead of implying
    # that a stable _original_code proves arbitrary active-code provenance.
    with _original_function(original.host_chain_dv2) as inner:
        _preprocess(inner)
        active = inner.__code__
        inner.__code__ = active.replace(co_name="unattested_transformed_name")
        host.check()
        assert inner.__code__ is not active
    host.check()


def test_precompile_active_original_code_identity_change_rejects(host):
    with _original_function(original.host_chain_dv2) as inner:
        assert "_original_code" not in vars(inner)
        inner.__code__ = inner.__code__.replace()
        with pytest.raises(_UnsupportedChain):
            host.check()
    host.check()
