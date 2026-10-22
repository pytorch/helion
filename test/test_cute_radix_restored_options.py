"""CPU schema, source and membership checks for restored radix options."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
import random
import struct
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_radix_selection import _payload_selection

from helion._compiler.cute.radix_selection import radix_schedule
from helion._compiler.cute.radix_selection_config import BOOLEAN_KEYS
from helion._compiler.cute.radix_selection_config import CONFIG_KEYS
from helion._compiler.cute.radix_selection_config import PIPELINE_CHOICES
from helion.exc import InvalidConfig
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import CodeType
    from typing_extensions import Self

    from helion.runtime.kernel import BoundKernel


_RESTORED_KEYS = (
    "cute_radix_resident_input",
    "cute_radix_boundary_early_exit",
    "cute_radix_defer_output",
    "cute_radix_partial_scan",
)
_FORMATS = {
    torch.float16: SimpleNamespace(width=16, cute_name="Float16", infinity=0x7C00),
    torch.bfloat16: SimpleNamespace(width=16, cute_name="BFloat16", infinity=0x7F80),
    torch.float32: SimpleNamespace(width=32, cute_name="Float32", infinity=0x7F800000),
}


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("HELION_AUTOTUNE_EFFORT", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_CONFIG_OVERRIDES", raising=False)
    with _forbid_cuda_work(), _mock_cuda_unavailable(), _target():
        yield


def _bind(dtype: torch.dtype = torch.bfloat16, cols: int = 33) -> BoundKernel[object]:
    args = (
        torch.empty((3, cols), dtype=dtype),
        torch.tensor([-(1 << 40), 1 << 40, cols], dtype=torch.int64),
        17,
    )
    return _cpu_bind(_payload_selection, args)


def _config(bound: BoundKernel[object], **changes: object) -> Config:
    result = bound.config_spec.default_config()
    result.config.update(
        cute_selection_algorithm="radix",
        cute_radix_threads=512,
        cute_radix_cluster_ctas=2,
        cute_radix_candidate_capacity=0,
        cute_radix_histogram_mode="direct",
        cute_radix_packet_bits=64,
        cute_radix_valid_domain=True,
        cute_radix_ticket_stores=True,
        **dict.fromkeys(_RESTORED_KEYS, True),
    )
    result.config.update(changes)
    return result


def test_restored_keys_and_packet_choices_survive_native_schema_and_roundtrip() -> None:
    bound = _bind()
    spec = bound.config_spec
    fields = spec._flat_fields()
    assert PIPELINE_CHOICES["cute_radix_packet_bits"] == (128, 256, 64)
    assert fields["cute_radix_packet_bits"].default() == 128
    assert fields["cute_radix_packet_bits"].search_values() == [128, 256, 64]
    default = spec.default_config()
    for key in _RESTORED_KEYS:
        assert key in BOOLEAN_KEYS and key in CONFIG_KEYS
        assert spec.supports_config_key(key)
        assert fields[key].search_values() == [False, True]
        assert default.get(key, False) is False
        malformed = _config(bound, **{key: 1})
        with pytest.raises(InvalidConfig, match="Boolean"):
            spec.normalized_config(malformed)
        legacy = Config.from_dict(default.config | {key: True})
        with pytest.raises(InvalidConfig, match="requires the radix"):
            spec.normalized_config(legacy)
    requested = _config(bound)
    normalized = spec.normalized_config(requested)
    generation = spec.create_config_generation()
    flat, decoded = generation.strict_config_pair(normalized)
    assert generation.unflatten(flat) == decoded == normalized
    assert decoded.get("cute_radix_packet_bits") == 64
    assert all(decoded.get(key) is True for key in _RESTORED_KEYS)


@pytest.mark.parametrize("dtype", _FORMATS)
@pytest.mark.parametrize("inverse", [False, True])
def test_resident_payload_and_key_only_paths_generate_typed_bounded_source(
    dtype: torch.dtype, inverse: bool
) -> None:
    bound = _bind(dtype, cols=257)
    config = _config(
        bound,
        cute_radix_cached_payload_inverse=inverse,
        cute_radix_ticket_stores=inverse,
    )
    normalized = bound.config_spec.normalized_config(config)
    graph = bound.config_spec.cute_radix_selection_graph
    assert graph is not None
    plan = radix_schedule(graph, normalized.config)
    assert plan.resident_input and plan.resident_keys_only is inverse
    assert plan.resident_elements == plan.source_packet_rounds * plan.packet
    assert plan.resident_elements * (1 if inverse else 2) <= 128
    allocations = {name for name, _, _ in plan.allocations}
    assert "primary_done" in allocations
    assert "coarse_prefix" not in allocations and "coarse_mask" not in allocations
    assert ("selected_indices" in allocations) is not inverse
    assert "candidate_keys" not in allocations
    source = bound.to_code(normalized)
    ast.parse(source)
    assert "resident_keys = cute.make_rmem_tensor" in source
    assert ("resident_values = cute.make_rmem_tensor" in source) is not inverse
    assert ("cached_payload_raw" in source) is inverse
    assert "coarse_prefix" not in source and "coarse_mask" not in source
    assert "primary_done[0] == 0" in source
    assert "barrier(barrier_id=1, number_of_threads=256)" in source
    assert "_helion_radix_consumer" not in source
    assert f"cutlass.{_FORMATS[dtype].cute_name}" in source


@pytest.mark.parametrize("inverse", [False, True])
def test_resident_input_rejects_register_budget_overflow(inverse: bool) -> None:
    bound = _bind(cols=32769)
    config = _config(
        bound,
        cute_radix_threads=256,
        cute_radix_cluster_ctas=1,
        cute_radix_cached_payload_inverse=inverse,
    )
    with pytest.raises(InvalidConfig, match="register budget"):
        bound.config_spec.normalized_config(config)


class _Signed(int):
    width = 32

    def __new__(cls, value: int) -> Self:
        raw = int(value) & ((1 << cls.width) - 1)
        sign = 1 << (cls.width - 1)
        return int.__new__(cls, raw - (1 << cls.width) if raw & sign else raw)


class _Signed16(_Signed):
    width = 16


class _Unsigned(int):
    width = 32

    def __new__(cls, value: int) -> Self:
        return int.__new__(cls, int(value) & ((1 << cls.width) - 1))

    def __xor__(self, other: int) -> _Unsigned:
        return type(self)(int(self) ^ int(other))

    def __and__(self, other: int) -> _Unsigned:
        return type(self)(int(self) & int(other))

    def __or__(self, other: int) -> _Unsigned:
        return type(self)(int(self) | int(other))

    def bitcast(self, target: type[Any]) -> Any:
        return (
            target.from_raw(int(self))
            if issubclass(target, _RawFloat)
            else target(self)
        )


class _Unsigned16(_Unsigned):
    width = 16


class _RawFloat:
    width = 32

    def __init__(self, value: _RawFloat | int) -> None:
        if type(value) is type(self):
            self.raw = value.raw
        else:
            assert value == 0
            self.raw = 0

    @classmethod
    def from_raw(cls, raw: int) -> _RawFloat:
        value = cls(0)
        value.raw = raw & ((1 << cls.width) - 1)
        return value

    def bitcast(self, target: type[Any]) -> Any:
        return target(self.raw)


class _RawFloat16(_RawFloat):
    width = 16


class _RawBFloat16(_RawFloat):
    width = 16


class _Pointer:
    def __init__(self, value: _RawFloat) -> None:
        self.value = value
        self.offset = 0
        self.loads = 0

    def __add__(self, offset: int) -> _Pointer:
        self.offset += int(offset)
        return self

    def load(self) -> _RawFloat:
        assert self.offset == 37 + 19
        self.loads += 1
        return self.value


_FLOAT_TYPES = {
    torch.float16: _RawFloat16,
    torch.bfloat16: _RawBFloat16,
    torch.float32: _RawFloat,
}
_CUTLASS = SimpleNamespace(
    Int16=_Signed16,
    Int32=_Signed,
    Uint16=_Unsigned16,
    Uint32=_Unsigned,
    Float16=_RawFloat16,
    BFloat16=_RawBFloat16,
    Float32=_RawFloat,
    Boolean=bool,
)


@lru_cache(maxsize=1)
def _emitters() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[1]
        / "helion/_compiler/cute/radix_selection_codegen.py"
    )
    names = {
        "_Writer",
        "_boolean",
        "_rank",
        "_primary_key",
        "_selection_key",
        "_reserve",
        "_synchronize_cluster",
        "_inverse_cached_payload",
        "_histogram_choose",
        "_compact_packet_work",
    }
    definitions = [
        node
        for node in ast.parse(path.read_text(), filename=str(path)).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == names
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, *definitions],
        type_ignores=[],
    )
    namespace = {"contextmanager": contextmanager, "_FORMATS": _FORMATS}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@lru_cache(maxsize=3)
def _payload_code(dtype: torch.dtype) -> tuple[CodeType, CodeType]:
    emit = _emitters()
    spec = SimpleNamespace(
        dtype=dtype,
        primary_width=_FORMATS[dtype].width,
        valid_domain=True,
        largest=True,
    )
    writer = emit["_Writer"]()
    emit["_primary_key"](writer, spec, "value", "column", "key")
    key = compile("\n".join(writer.lines), "<emitted-radix-key>", "exec")
    writer = emit["_Writer"]()
    emit["_inverse_cached_payload"](writer, spec)
    inverse = compile("\n".join(writer.lines), "<emitted-radix-inverse>", "exec")
    return key, inverse


def _patterns(dtype: torch.dtype) -> list[int]:
    kind = _FORMATS[dtype]
    sign = 1 << (kind.width - 1)
    normal = (
        0x0400
        if dtype == torch.float16
        else (0x0080 if kind.width == 16 else 0x00800000)
    )
    positive = [0, 1, normal - 1, normal, normal + 1, kind.infinity - 1, kind.infinity]
    positive += [kind.infinity + 1, kind.infinity + 3, sign - 1]
    rng = random.Random(391)
    return [*positive, *(raw | sign for raw in positive)] + [
        rng.randrange(1 << kind.width) for _ in range(96)
    ]


def _canonical_f32_rank(dtype: torch.dtype, raw: int) -> int:
    if dtype == torch.bfloat16:
        raw32 = raw << 16
    elif dtype == torch.float16:
        value = struct.unpack("<e", struct.pack("<H", raw))[0]
        raw32 = struct.unpack("<I", struct.pack("<f", value))[0]
    else:
        raw32 = raw
    magnitude = raw32 & 0x7FFFFFFF
    if magnitude > 0x7F800000:
        return 0x7FFFFFFF
    return -magnitude if raw32 & 0x80000000 else magnitude


@pytest.mark.parametrize("dtype", _FORMATS)
def test_emitted_resident_key_inverse_preserves_special_payload_bits_and_rank(
    dtype: torch.dtype,
) -> None:
    key_code, inverse_code = _payload_code(dtype)
    kind = _FORMATS[dtype]
    state = {
        "cutlass": _CUTLASS,
        "column": 0,
        "selection_length": 1,
        "output_source_column": 19,
        "input_row_base": 37,
        "vec": 0,
    }
    ordered = []
    for raw in _patterns(dtype):
        value = _FLOAT_TYPES[dtype].from_raw(raw)
        state["value"] = value
        exec(key_code, state)
        key = int(state["key"])
        pointer = _Pointer(value)
        state.update(input_keys=[key], x=SimpleNamespace(iterator=pointer))
        exec(inverse_code, state)
        assert state["cached_payload_raw"].raw == raw, (dtype, hex(raw), hex(key))
        magnitude = raw & ((1 << (kind.width - 1)) - 1)
        assert pointer.loads == int(magnitude == 0 or magnitude > kind.infinity)
        ordered.append((key, _canonical_f32_rank(dtype, raw)))
    for left, right in pairwise(sorted(ordered)):
        assert left[1] == right[1] if left[0] == right[0] else left[1] < right[1]


@pytest.mark.parametrize("dtype", _FORMATS)
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("defer", [False, True])
def test_emitted_boundary_and_compaction_preserve_unordered_membership(
    dtype: torch.dtype, early: bool, defer: bool
) -> None:
    emit = _emitters()
    width = _FORMATS[dtype].width
    spec = SimpleNamespace(
        plan=SimpleNamespace(
            partial_scan=False,
            boundary_early_exit=early,
            defer_output=defer,
            resident_input=True,
        ),
        cluster=1,
        threads=256,
        ticket_stores=False,
    )
    writer = emit["_Writer"]()
    emit["_histogram_choose"](
        writer,
        spec,
        prefix="primary_prefix",
        mask="primary_mask",
        word="cutlass.Uint32",
        shift="primary_shift",
    )
    chosen = next(
        node
        for node in ast.walk(ast.parse("\n".join(writer.lines)))
        if isinstance(node, ast.If)
        and all(
            term in ast.unparse(node.test)
            for term in ("bin_count > 0", "greater_count", "pass_remaining")
        )
    )
    choose_code = compile(
        ast.Module(body=[chosen], type_ignores=[]), "<emitted-radix-boundary>", "exec"
    )
    writer = emit["_Writer"]()
    emit["_compact_packet_work"](writer, spec, False)
    predicates = ast.parse("\n".join(writer.lines)).body
    end = next(
        index
        for index, node in enumerate(predicates)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "above_members"
    )
    compact_code = compile(
        ast.Module(body=predicates[:end], type_ignores=[]),
        "<emitted-radix-compaction>",
        "exec",
    )
    key_code, _inverse_code = _payload_code(dtype)
    special_keys = []
    for raw in _patterns(dtype)[:20]:
        state = {"cutlass": _CUTLASS, "value": _FLOAT_TYPES[dtype].from_raw(raw)}
        exec(key_code, state)
        special_keys.append(int(state["key"]))
    prefix = 191 << (width - 8)
    datasets = [
        list(range(8)),
        [(1 << width) - 1] * 8,
        [prefix | 17] * 6 + [prefix | 16] * 3 + [193 << (width - 8) | 1],
        special_keys,
    ]
    for keys in datasets:
        for k in {1, len(keys), max(1, len(keys) // 2)}:
            state = {
                "cutlass": SimpleNamespace(Int32=int, Uint32=int, Boolean=bool),
                "primary_prefix": [0],
                "primary_mask": [0],
                "remaining": [k],
                "primary_done": [0],
                "coarse_prefix": [0],
                "coarse_mask": [0],
            }
            above = []
            for round_no in range(width // 8):
                if early and state["primary_done"][0]:
                    break
                prefix = state["primary_prefix"][0]
                mask = state["primary_mask"][0]
                remaining = state["remaining"][0]
                if round_no == 1 and not defer:
                    state["coarse_prefix"][0] = prefix
                    state["coarse_mask"][0] = mask
                    above.extend(i for i, key in enumerate(keys) if key & mask > prefix)
                shift = (width // 8 - 1 - round_no) * 8
                histogram = [0] * 256
                for key in keys:
                    if key & mask == prefix:
                        histogram[key >> shift & 255] += 1
                for digit, count in enumerate(histogram):
                    state.update(
                        thread=digit,
                        bin_count=count,
                        greater_count=sum(histogram[digit + 1 :]),
                        pass_prefix=prefix,
                        pass_mask=mask,
                        pass_remaining=remaining,
                        primary_shift=shift,
                    )
                    exec(choose_code, state)
            equal = []
            all_keys = keys + [(1 << width) - 1] * 3
            for index, key in enumerate(all_keys):
                state.update(
                    input_keys=[key],
                    input_members=[index < len(keys)],
                    vec=0,
                    primary_boundary=state["primary_prefix"][0],
                    boundary_complete=bool(state["primary_done"][0]),
                )
                exec(compact_code, state)
                if state["take_above"]:
                    above.append(index)
                if state["take_equal"]:
                    equal.append(index)
            selected = above + equal[: state["remaining"][0]]
            assert len(selected) == k and len(set(selected)) == k
            actual = sorted((all_keys[index] for index in selected), reverse=True)
            assert actual == sorted(keys, reverse=True)[:k]
