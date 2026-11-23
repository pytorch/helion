"""CPU codegen for the ``cute_wide_packet_hints`` knob: L2 policies on
32-byte packets.

``load_eviction_policies`` values ``l2_last`` / ``l1_l2_first`` /
``l1_l2_last`` reach 16-, 8- and 4-byte packets through the inline-PTX
helpers of ``l2_policy.py``.  A 32-byte packet (``cute_vector_widths`` 8 for
fp32, 16 for bf16/fp16; admitted on SM100+) dropped the hint: the helper
table had no 32-byte row.  The row exists now (one ``ld.global...v8.b32``
with the policy) behind the default-off ``cute_wide_packet_hints`` config
knob, so recorded configs render exactly as before; the knob is also inert
on a toolchain without the 256-bit hinted load (pre-SM100 or CUDA < 13).
The GPU numerics are at the end of the module.
"""

from __future__ import annotations

import ast
import copy
import importlib
import logging
import re
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from helion._compiler.cute.cache_policy_loads import PACKET_LOAD_HELPERS
from helion._compiler.cute.cutedsl_compat import readonly_packet_load_supported
from helion._compiler.cute.persistent_branch_vec import (
    vectorize_branch_local_persistent_fragments,
)
from helion._compiler.cute.target_facts import TargetFacts
from helion._compiler.triton.backend import TritonBackend
from helion._testing import DEVICE
from helion._testing import import_example
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.metrics import AutotuneMetrics
import helion.language as hl
from helion.runtime.config_cache import ConfigCache

rms_norm_fwd = import_example("rms_norm").rms_norm_fwd

# Source checks, in the CPU lane: CUDA hidden and native compilation
# forbidden; no device facts are modelled.  The GPU numerics at the end of
# the module are marked ``cuda``.
pytestmark = [pytest.mark.cpu_only, pytest.mark.cpu_render(TargetFacts())]

_HELPER = {
    "l2_last": "_cute_load_l2_evict_last",
    "l1_l2_first": "_cute_load_l1_l2_evict_first",
    "l1_l2_last": "_cute_load_l1_l2_evict_last",
}


def _row_kernel() -> helion.Kernel:
    return helion.kernel(
        rms_norm_fwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[exc.TensorOperationInWrapper, exc.FastMathUnused],
    )


def _bind_rows(
    dtype: torch.dtype, *, capability: tuple[int, int] = (10, 0)
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    # The arguments stay alive with the bound kernel: the alignment proof
    # reads the live inputs' metadata through weak references.
    arguments = (torch.empty((256, 1024), dtype=dtype), torch.empty(1024, dtype=dtype))
    return cpu_bind(_row_kernel(), arguments, capability=capability), arguments


def _row_config(
    bound: Any, *, threads: int, vec: int, policies: list[str], **extra: object
) -> helion.Config:
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    return spec.normalized_config(
        helion.Config(
            block_sizes=[1 for _ in spec.block_sizes.valid_block_ids()],
            num_threads=[
                threads if block_id == reduction else 0
                for block_id in spec.num_threads.valid_block_ids()
            ],
            cute_vector_widths=[
                vec if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            load_eviction_policies=policies,
            **extra,  # pyrefly: ignore [bad-argument-type]
        )
    )


def _kernel_body(code: str) -> str:
    return code.split("def _helion_")[1].split("\ndef ")[0]


# One 32-byte packet per thread of a 1024-wide row.
_ROW_PACKETS = [
    (torch.float32, 8, 128),
    (torch.bfloat16, 16, 64),
    (torch.float16, 16, 64),
]


@pytest.mark.parametrize("policy", ["l2_last", "l1_l2_first", "l1_l2_last"])
@pytest.mark.parametrize("dtype,vec,threads", _ROW_PACKETS)
def test_row_packets_take_the_32_byte_helper_under_the_knob(
    dtype: torch.dtype, vec: int, threads: int, policy: str
) -> None:
    bound, _arguments = _bind_rows(dtype)
    policies = [policy, policy]
    off = _kernel_body(
        bound.to_code(_row_config(bound, threads=threads, vec=vec, policies=policies))
    )
    on = _kernel_body(
        bound.to_code(
            _row_config(
                bound,
                threads=threads,
                vec=vec,
                policies=policies,
                cute_wide_packet_hints=True,
            )
        )
    )
    element = "Uint16" if dtype.itemsize == 2 else "Uint32"
    vector_type = f"ir.VectorType.get([{vec}], cutlass.{element}.mlir_type)"
    # Default: the hint is dropped on the 32-byte packet, as it always was.
    assert "_32b(" not in off
    assert off.count("cute.arch.load(x.iterator") == 1
    assert off.count("cute.arch.load(weight.iterator") == 1
    assert off.count(vector_type) == 2
    # Knob: the same two packets through the 32-byte helper of the policy.
    assert f"{_HELPER[policy]}_32b(x.iterator" in on
    assert f"{_HELPER[policy]}_32b(weight.iterator" in on
    assert "cute.arch.load(" not in on
    assert on.count(vector_type) == 2
    # Only the two load calls change.
    assert on.replace(f"{_HELPER[policy]}_32b(", "cute.arch.load(") == off


@pytest.mark.parametrize("dtype,vec,threads", [(torch.bfloat16, 8, 128)])
def test_sixteen_byte_packets_ignore_the_knob(
    dtype: torch.dtype, vec: int, threads: int
) -> None:
    bound, _arguments = _bind_rows(dtype)
    policies = ["l1_l2_last", "l2_last"]
    off = bound.to_code(_row_config(bound, threads=threads, vec=vec, policies=policies))
    on = bound.to_code(
        _row_config(
            bound,
            threads=threads,
            vec=vec,
            policies=policies,
            cute_wide_packet_hints=True,
        )
    )
    assert "_cute_load_l1_l2_evict_last(x.iterator" in _kernel_body(off)
    assert "_32b(" not in on
    assert on == off


@pytest.mark.parametrize(
    "cuda_version,capability", [("12.8", (10, 0)), (None, (10, 0)), ("13.0", (9, 0))]
)
def test_knob_is_inert_without_the_256_bit_hinted_load(
    cuda_version: str | None, capability: tuple[int, int]
) -> None:
    # PTX ``ld.global...v8.b32`` needs SM100+ and CUDA 13: elsewhere the
    # knob renders exactly the knob-off code (the ``nc_l2_first`` gate).
    bound, _arguments = _bind_rows(torch.float32, capability=capability)
    policies = ["l2_last", "l2_last"]
    with patch("torch.version.cuda", cuda_version):
        off = bound.to_code(_row_config(bound, threads=128, vec=8, policies=policies))
        on = bound.to_code(
            _row_config(
                bound,
                threads=128,
                vec=8,
                policies=policies,
                cute_wide_packet_hints=True,
            )
        )
    assert "_32b(" not in on
    assert on == off


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mul1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * y[tile]
    return out


@pytest.mark.parametrize("dtype,vec", [(torch.float32, 8), (torch.bfloat16, 16)])
def test_tile_hoist_packets_take_the_32_byte_helper_under_the_knob(
    dtype: torch.dtype, vec: int
) -> None:
    # The grid tile hoist (``_cute_register_tile_unroll_vec_load``): one
    # hinted 32-byte packet per thread and input (the block holds ``vec``
    # lanes per thread), each policy through its own helper.
    arguments = (torch.empty(2**16, dtype=dtype), torch.empty(2**16, dtype=dtype))
    bound = cpu_bind(_mul1d, arguments)
    spec = bound.config_spec

    def config(**extra: object) -> helion.Config:
        return spec.normalized_config(
            helion.Config(
                block_sizes=[256 * vec],
                num_threads=[256],
                cute_vector_widths=[vec],
                load_eviction_policies=["l2_last", "l1_l2_first"],
                **extra,  # pyrefly: ignore [bad-argument-type]
            )
        )

    off = _kernel_body(bound.to_code(config()))
    on = _kernel_body(bound.to_code(config(cute_wide_packet_hints=True)))
    assert "_32b(" not in off
    assert off.count("cute.arch.load(") == 2
    assert on.count("_cute_load_l2_evict_last_32b(x.iterator") == 1
    assert on.count("_cute_load_l1_l2_evict_first_32b(y.iterator") == 1
    assert "cute.arch.load(" not in on


def test_branch_local_persistent_packets_take_the_32_byte_helper() -> None:
    # The late branch-local vectorizer is the second helper-selection site.
    source = """
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    value = _helion_persistent_branch_vec_load(7, 8, 'cutlass.Float32', '__l1_l2_last__', x.iterator + lane_index, (x.iterator + lane_index).load())
    acc = acc + value
"""
    plain = ast.unparse(
        ast.Module(
            body=vectorize_branch_local_persistent_fragments(ast.parse(source).body),
            type_ignores=[],
        )
    )
    hinted = ast.unparse(
        ast.Module(
            body=vectorize_branch_local_persistent_fragments(
                ast.parse(source).body, wide_packet_hints=True
            ),
            type_ignores=[],
        )
    )
    packet = "(x.iterator + (lane_base + cutlass.Int32(0)), ir.VectorType.get([8], cutlass.Uint32.mlir_type))"
    assert f"_persistent_branch_vec_0 = cute.arch.load{packet}" in plain
    assert (
        f"_persistent_branch_vec_0 = _cute_load_l1_l2_evict_last_32b{packet}" in hinted
    )
    assert (
        hinted.replace("_cute_load_l1_l2_evict_last_32b(", "cute.arch.load(") == plain
    )


def _set_policies(config: helion.Config) -> list[int]:
    policies = config.config["load_eviction_policies"]
    assert isinstance(policies, list)
    return [index for index, policy in enumerate(policies) if policy]


@pytest.mark.parametrize(
    "vec,threads,policies,kept,renders",
    [
        # 32-byte packets have no hinted L2 load without the knob: both
        # policies are dead, and one render of the cleared config says so.
        (8, 128, ["l2_last", "l2_last"], [], 1),
        # The L1 priority reaches the 32-byte packet; the L2 policy does not.
        (8, 128, ["last", "l2_last"], [0], 3),
        # 16-byte packets take the hinted load: both policies are live.
        (4, 256, ["l2_last", "l2_last"], [0, 1], 3),
        # One live policy: the cleared render already answers for its slot.
        (4, 256, ["l2_last", ""], [0], 1),
    ],
)
def test_selected_config_keeps_only_the_policies_its_code_carries(
    vec: int, threads: int, policies: list[str], kept: list[int], renders: int
) -> None:
    bound, _arguments = _bind_rows(torch.float32)
    selected = _row_config(bound, threads=threads, vec=vec, policies=policies)
    code = bound.to_code(selected)
    asked: list[helion.Config] = []

    def same_source(candidate: helion.Config) -> bool:
        asked.append(candidate)
        return bound.to_code(candidate) == code

    simplified = CuteBackend().simplified_selected_config(
        bound.config_spec, selected, same_source
    )

    assert _set_policies(simplified) == kept
    assert len(asked) == renders
    # The same kernel, and nothing but the policies changed.
    assert bound.to_code(simplified) == code
    assert {
        key: value
        for key, value in simplified.config.items()
        if key != "load_eviction_policies"
    } == {
        key: value
        for key, value in selected.config.items()
        if key != "load_eviction_policies"
    }
    if kept == _set_policies(selected):
        assert simplified is selected
    # Every policy that stays is live: resetting it changes the code.
    for index in kept:
        values = list(simplified.config["load_eviction_policies"])  # type: ignore[call-overload]
        values[index] = ""
        reset = helion.Config(**{**simplified.config, "load_eviction_policies": values})  # type: ignore[arg-type]
        assert bound.to_code(reset) != code


def _selecting_search(
    bound: Any, arguments: Any, selected: helion.Config
) -> tuple[PopulationBasedSearch, PopulationMember]:
    """A bare search over the real bound kernel that measured ``selected``;
    returns it with the measured member."""
    search = object.__new__(PopulationBasedSearch)
    search.config_spec = bound.config_spec
    search.config_gen = bound.config_spec.create_config_generation()
    search.kernel = bound
    search.settings = copy.copy(bound.settings)
    search.args = arguments
    search.benchmark_provider = SimpleNamespace(settles_source_aliases=lambda: True)  # type: ignore[assignment]
    search.log = Mock()
    search._kernels = {}
    search._benchmarked_members = ConfigCache()
    search._autotune_metrics = AutotuneMetrics()
    measured = PopulationMember(
        bound.compile_config(selected, allow_print=False),
        [1.0],
        search.config_gen.flatten(selected),
        selected,
        status="ok",
    )
    search._selected_member = measured
    return search, measured


def test_search_returns_the_selected_kernel_without_its_dead_policies() -> None:
    """The search side with a real bound kernel: candidates are rendered in
    process and compared with the measured kernel by generated-source
    hash."""
    bound, arguments = _bind_rows(torch.float32)
    selected = _row_config(bound, threads=128, vec=8, policies=["last", "l2_last"])
    search, measured = _selecting_search(bound, arguments, selected)

    returned = search._simplified_selected_config(selected)

    assert _set_policies(returned) == [0]
    assert bound.to_code(returned) == bound.to_code(selected)
    # The returned config stays linked to the kernel that was measured.
    assert search._selected_member is not measured
    assert search._selected_member.config == returned
    assert search._selected_member.fn is measured.fn
    assert search._selected_member.perfs == [1.0]
    (message,) = [call.args[0] for call in search.log.call_args_list]
    assert message.startswith("Returning a simpler config")
    # A selection whose policies are all live comes back as it is.
    live = _row_config(bound, threads=256, vec=4, policies=["l2_last", "l2_last"])
    search, _measured = _selecting_search(bound, arguments, live)
    assert search._simplified_selected_config(live) is live


class _StderrHandler(logging.Handler):
    """Writes records to whatever ``sys.stderr`` is when they are emitted,
    as a console handler of the application would."""

    def emit(self, record: logging.LogRecord) -> None:
        sys.stderr.write(self.format(record) + "\n")
        sys.stderr.flush()


@pytest.mark.parametrize("ignore_errors", [True, False])
def test_a_proposal_that_fails_to_render_leaves_the_console_alone(
    ignore_errors: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """A proposal whose render raises inside the real ``compile_config``:
    the runtime logs the failure with a traceback before it re-raises.  The
    search captures that output for a candidate, and the step that renders
    the proposals does the same, so under ``autotune_ignore_errors`` nothing
    is printed, and otherwise only the step's own warning is logged.  The
    proposals after the failing one are still tried."""
    bound, arguments = _bind_rows(torch.float32)
    selected = _row_config(bound, threads=128, vec=8, policies=["last", "l2_last"])
    search, _measured = _selecting_search(bound, arguments, selected)
    search.settings.autotune_ignore_errors = ignore_errors
    render = bound.to_triton_code

    def failing(config: helion.Config, **kwargs: Any) -> str:
        if not _set_policies(config):
            raise RuntimeError("the cleared config does not render")
        return render(config, **kwargs)

    runtime_log = logging.getLogger("helion.runtime.kernel")
    handler = _StderrHandler()
    runtime_log.addHandler(handler)
    try:
        with patch.object(bound, "to_triton_code", side_effect=failing):
            returned = search._simplified_selected_config(selected)
    finally:
        runtime_log.removeHandler(handler)

    assert _set_policies(returned) == [0]
    assert "codegen error" not in capsys.readouterr().err
    warnings = [call.args[0] for call in search.log.warning.call_args_list]
    if ignore_errors:
        assert warnings == []
    else:
        (warning,) = warnings
        assert warning.startswith("Not taking ")


def test_knob_is_a_cute_boolean() -> None:
    config = helion.Config(cute_wide_packet_hints=True)
    assert config.config == {"cute_wide_packet_hints": True}
    assert config.cute_wide_packet_hints is True
    assert helion.Config().cute_wide_packet_hints is False
    assert helion.Config.from_json(config.to_json()).cute_wide_packet_hints is True
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0))
    spec.normalize(config)
    assert config.config["cute_wide_packet_hints"] is True
    # Absent stays absent (the search layout does not carry the knob).
    absent: dict[str, object] = {}
    spec.normalize(absent)
    assert "cute_wide_packet_hints" not in absent
    with pytest.raises(exc.InvalidConfig, match="must be a boolean"):
        spec.normalize({"cute_wide_packet_hints": 1})
    repaired = {"cute_wide_packet_hints": "yes"}
    spec.normalize(repaired, _fix_invalid=True)
    assert repaired["cute_wide_packet_hints"] is False
    with pytest.raises(exc.InvalidConfig, match="Unsupported config keys"):
        ConfigSpec(backend=TritonBackend()).normalize({"cute_wide_packet_hints": True})


_IMPORT_AS = re.compile(r"^from (\S+) import (\w+) as (\w+)$")


def test_helper_table_rows_match_their_device_helper_width() -> None:
    # Device-module invariant: every L2-policy row of ``PACKET_LOAD_HELPERS``
    # is bound by the backend to a helper whose inline asm loads exactly that
    # many bytes (``.v{bytes // 4}.b32``; the 4-byte form is the scalar
    # ``.b32``) and returns the requested vector type.
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func

    imports = CuteBackend().library_imports
    rows = [
        (helper.packet_bytes, helper.name)
        for helper in PACKET_LOAD_HELPERS
        if helper.name in _CUTE_CACHE_LOAD_HELPER_NAMES
    ]

    class Address:
        def __init__(self, value: Any) -> None:
            self.value = value

        def toint(self, **kwargs: Any) -> Address:
            return self

        def ir_value(self, **kwargs: Any) -> Any:
            return self.value

    with ir.Context(), ir.Location.unknown():
        for packet_bytes, name in rows:
            match = _IMPORT_AS.match(imports[name])
            assert match is not None, imports[name]
            module_name, attribute, alias = match.groups()
            assert alias == name
            helper = getattr(importlib.import_module(module_name), attribute)
            module = ir.Module.create()
            vector_type = ir.VectorType.get(
                [packet_bytes // 2], cutlass.Uint16.mlir_type
            )
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    name, ir.FunctionType.get([cutlass.Uint64.mlir_type], [vector_type])
                )
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                value = helper(Address(block.arguments[0]), vector_type)
                assert value.type == vector_type
                func.ReturnOp([value])
            module.operation.verify()
            text = str(module)
            words = packet_bytes // 4
            form = f".v{words}.b32" if words > 1 else ".b32 $0, [$1]"
            assert form in text, (name, packet_bytes, text)
            # One load of exactly this width: no other vector width appears.
            for other in (2, 4, 8):
                if other != words:
                    assert f".v{other}.b32" not in text, (name, packet_bytes)
            assert text.count("llvm.inline_asm has_side_effects") == 1


# GPU numerics for ``cute_wide_packet_hints``: a 32-byte packet carrying an L2
# policy through the SM100 ``_32b`` helpers loads every element exactly once.  A cache policy
# changes only the opcode of a proven packet, so every hinted render must match
# the knob-off render bit for bit.


_HELPER_32B = {
    "l2_last": "_cute_load_l2_evict_last_32b",
    "l1_l2_first": "_cute_load_l1_l2_evict_first_32b",
    "l1_l2_last": "_cute_load_l1_l2_evict_last_32b",
}


def _require_wide_hinted_loads() -> None:
    if not readonly_packet_load_supported(
        torch.cuda.get_device_capability(), torch.version.cuda
    ):
        pytest.skip("ld.global...v8.b32 needs SM100+ and CUDA 13")


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0), reason="32-byte packets need SM100+")
@pytest.mark.parametrize("policy", ["l2_last", "l1_l2_first", "l1_l2_last"])
@pytest.mark.parametrize(
    "dtype,vec,threads",
    [(torch.float32, 8, 128), (torch.bfloat16, 16, 64), (torch.float16, 16, 64)],
)
def test_hinted_32_byte_row_packets_match_the_plain_render(
    policy: str, dtype: torch.dtype, vec: int, threads: int
) -> None:
    _require_wide_hinted_loads()
    torch.manual_seed(0)
    x = torch.randn(256, 1024, dtype=dtype, device=DEVICE)
    weight = torch.randn(1024, dtype=dtype, device=DEVICE)
    kernel = helion.kernel(
        rms_norm_fwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    bound = kernel.bind((x, weight))
    policies = [policy, policy]
    hinted = _row_config(
        bound, threads=threads, vec=vec, policies=policies, cute_wide_packet_hints=True
    )
    plain = _row_config(bound, threads=threads, vec=vec, policies=policies)
    body = _kernel_body(bound.to_code(hinted))
    assert f"{_HELPER_32B[policy]}(x.iterator" in body
    assert f"{_HELPER_32B[policy]}(weight.iterator" in body
    assert "_32b(" not in _kernel_body(bound.to_code(plain))
    out, inv = bound.compile_config(hinted)(x, weight)
    plain_out, plain_inv = bound.compile_config(plain)(x, weight)
    torch.testing.assert_close(out, plain_out, atol=0, rtol=0)
    torch.testing.assert_close(inv, plain_inv, atol=0, rtol=0)
    xf = x.float()
    ref_inv = torch.rsqrt(xf.pow(2).mean(-1) + 1e-5)
    torch.testing.assert_close(inv, ref_inv.reshape(-1, 1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        out, (xf * ref_inv[:, None] * weight.float()).to(dtype), atol=1e-2, rtol=1e-2
    )


@pytest.mark.cuda
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((10, 0), reason="32-byte packets need SM100+")
@pytest.mark.parametrize("dtype,vec", [(torch.float32, 8), (torch.bfloat16, 16)])
def test_hinted_32_byte_tile_packets_load_every_element(
    dtype: torch.dtype, vec: int
) -> None:
    _require_wide_hinted_loads()
    torch.manual_seed(0)
    x = torch.randn(2**16, dtype=dtype, device=DEVICE)
    y = torch.randn_like(x)
    bound = _mul1d._bind_isolated((x, y))
    config = helion.Config(
        block_sizes=[256 * vec],
        num_threads=[256],
        cute_vector_widths=[vec],
        load_eviction_policies=["l2_last", "l1_l2_first"],
        cute_wide_packet_hints=True,
    )
    body = _kernel_body(bound.to_code(config))
    assert "_cute_load_l2_evict_last_32b(x.iterator" in body
    assert "_cute_load_l1_l2_evict_first_32b(y.iterator" in body
    out = bound.compile_config(config)(x, y)
    torch.testing.assert_close(out, x * y, atol=0, rtol=0)
