from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import types
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache
from torch.utils._pytree import tree_leaves

from test.test_cute_repro_settings import _case
from test.test_cute_structural_aot import _function
from test.test_cute_structural_aot import _module
from test.test_cute_structural_cache import _cpu_only  # noqa: F401

import helion
from helion._testing import skipUnlessBackends
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_compile import generate_standalone_file
from helion.autotuner.aot_structural_export import _export_classifiers
from helion.autotuner.aot_structural_export import export_runtime_guards
from helion.autotuner.aot_structural_policy import attach_model_policy
from helion.autotuner.aot_structural_policy import policy_path
from helion.autotuner.base_search import BaseSearch
import helion.language as hl
from helion.runtime.cute_structural_config import StructuralPolicyError
from helion.runtime.cute_structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute_structural_policy import CuteStructuralPolicy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from typing import Any


@pytest.fixture(autouse=True)
def _shared_budget() -> Iterator[None]:
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=232448,
    ):
        yield


def _relocate(function: Callable[..., Any], tmp_path: Path) -> Callable[..., Any]:
    source = tmp_path / "original.py"
    source.write_bytes(Path(function.__code__.co_filename).read_bytes())
    copied = types.FunctionType(
        function.__code__.replace(co_filename=str(source)),
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    copied.__annotations__ = function.__annotations__.copy()
    return copied


def _model(
    name: str, configs: list[helion.Config], policy: CuteStructuralPolicy, size: int
) -> str:
    raw = f"def key_{name}(*args):\n    return int(args[0].shape[0] > {size})\ndef autotune_{name}(*args):\n    return {{}}\n"
    return attach_model_policy(raw, {name: configs}, policy)


def _metadata(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), value.stride())
    if isinstance(value, (tuple, list)):
        return tuple(_metadata(item) for item in value)
    return value


def _call(
    fn: Callable[..., Any], args: tuple[object, ...]
) -> tuple[object, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    def hold(
        kernel: Callable[..., Any], grid: object, *values: object, **kwargs: object
    ) -> None:
        calls.append(
            {
                "kernel": kernel,
                "source": inspect.getsource(kernel),
                "grid": grid,
                "arguments": _metadata(values),
                "kwargs": kwargs,
                "constants": {
                    name: value
                    for name, value in kernel.__globals__.items()
                    if name.startswith("_BLOCK_SIZE_")
                },
            }
        )

    result = fn(*args, _launcher=hold)
    return result, calls


@pytest.mark.parametrize("setting", CUTE_STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("static", [False, True])
@skipUnlessBackends(["cute"])
def test_actual_public_export_keeps_all_structural_source_closures(
    setting: str,
    static: bool,
    tmp_path: Path,
) -> None:
    original, args, marker = _case(setting)
    function = _relocate(original, tmp_path)
    policy = CuteStructuralPolicy(**{setting: True})
    kernel = helion.aot_kernel(
        function,
        backend="cute",
        static_shapes=static,
        cute_structural_policy=policy,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    configs = [bound.config_spec.default_config()]
    configs.append(copy.deepcopy(configs[0]))
    configs[1].block_sizes[0] *= 2
    model = tmp_path / "model.py"
    model.write_text(_model(kernel.name, configs, policy, args[0].shape[0]))
    cache = AOTAutotuneCache(BaseSearch(bound, args))
    generated = []
    original_codegen = bound.to_triton_code

    def capture_codegen(config: helion.Config) -> str:
        code = original_codegen(config)
        generated.append(code)
        return code

    with (
        patch.object(cache, "_find_heuristic_file", return_value=model),
        patch.object(bound, "to_triton_code", side_effect=capture_codegen),
    ):
        cache._maybe_run_compile()
    path = policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    exported = _module(path)
    direct = PyCodeCache.load(generated[0])
    result, before = _call(getattr(direct, kernel.name), args)
    exported_result, after = _call(getattr(exported, kernel.name), args)
    assert _metadata(result) == _metadata(exported_result)
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(tree_leaves(result), tree_leaves(exported_result), strict=True)
    )
    assert len(before) == len(after) >= 1
    for a, b in zip(before, after, strict=True):
        assert {k: v for k, v in a.items() if k != "kernel"} == {
            k: v for k, v in b.items() if k != "kernel"
        }
    prefix = "_helion_aot"
    sources = vars(exported)[f"{prefix}_sources"]
    assert tuple(generated) == sources
    assert vars(exported)[f"{prefix}_source_hashes"] == tuple(
        hashlib.sha256(code.encode()).hexdigest() for code in sources
    )
    modules = vars(exported)[f"{prefix}_modules"]
    assert Path(modules[0].__file__).read_text() == sources[0]
    # Source module identity includes the explicit policy even if two policies
    # happen to generate identical code. Child sources retain their own hashes.
    assert modules[0] is PyCodeCache.load(sources[0], extra=policy.identity())
    assert (
        modules[0]
        is not PyCodeCache.load(sources[0], extra=CuteStructuralPolicy().identity())
        if policy != CuteStructuralPolicy()
        else True
    )
    if not static:
        assert len(sources) == 2
        second_direct = PyCodeCache.load(sources[1])
        _, direct_calls = _call(getattr(second_direct, kernel.name), args)
        _, linked_calls = _call(vars(exported)[f"{prefix}_functions"][1], args)
        assert [call["constants"] for call in direct_calls] == [
            call["constants"] for call in linked_calls
        ]
        for module in modules:
            wrapper = getattr(module, kernel.name)
            if "_helion_cute_kernels" in vars(wrapper):
                assert (
                    tuple(call["kernel"] for call in _call(wrapper, args)[1])
                    == wrapper._helion_cute_kernels
                )
    fresh = tuple(
        torch.empty_strided(arg.shape, arg.stride(), dtype=arg.dtype)
        if isinstance(arg, torch.Tensor)
        else arg
        for arg in args
    )
    _call(getattr(exported, kernel.name), fresh)
    altered = list(fresh)
    altered[0] = torch.empty(
        (fresh[0].shape[0] * 2, *fresh[0].shape[1:]), dtype=fresh[0].dtype
    )
    requires_bound = bool(
        bound.env.specialized_vars
        or bound.env.specialized_strides
        or bound.env.tensor_descriptor_layout_guards
        or any(
            extractor.fact == "input_tensor_metadata"
            for extractor in bound._compiler_seed_specialization_extractors
        )
    )
    if static or requires_bound:
        hold = Mock(side_effect=AssertionError("unsupported child invoked"))
        with pytest.raises(ValueError, match="No standalone variant"):
            getattr(exported, kernel.name)(*altered, _launcher=hold)
        hold.assert_not_called()
    else:
        _, selected = _call(getattr(exported, kernel.name), tuple(altered))
        _, expected = _call(vars(exported)[f"{prefix}_functions"][1], tuple(altered))
        assert selected == expected
    (tmp_path / "closure-result.json").write_text(
        json.dumps(
            {
                "setting": setting,
                "static": static,
                "launches": len(after),
                "requires_bound": requires_bound,
                "sources": list(vars(exported)[f"{prefix}_source_hashes"]),
                "output": str(path),
                "marker": marker,
            },
            indent=2,
        )
    )


@skipUnlessBackends(["cute"])
def test_dynamic_specialized_export_accumulates_exact_metadata_groups(
    tmp_path: Path,
) -> None:
    original, args, _ = _case("cute_materialize_transformed_operands")
    function = _relocate(original, tmp_path)
    policy = CuteStructuralPolicy(cute_materialize_transformed_operands=True)
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=False, cute_structural_policy=policy
    )
    bound = kernel._bind_isolated(args)
    config = bound.config_spec.default_config()
    model = tmp_path / "model.py"
    code = f"def key_{kernel.name}(*args):\n    return 0\ndef autotune_{kernel.name}(*args):\n    return {{}}\n"
    model.write_text(attach_model_policy(code, {kernel.name: [config]}, policy))
    inputs = [args, (torch.empty((32, 40), dtype=torch.bfloat16), args[1])]
    for values in inputs:
        current = kernel._bind_isolated(values)
        cache = AOTAutotuneCache(BaseSearch(current, values))
        with patch.object(cache, "_find_heuristic_file", return_value=model):
            cache._maybe_run_compile()
    path = policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    exported = _module(path)
    assert len(exported._helion_aot_groups) == 2
    assert all(key[0] == "exact" for key in exported._helion_aot_groups)
    for values in inputs:
        _call(getattr(exported, kernel.name), values)
    hold = Mock()
    with pytest.raises(ValueError, match="No standalone variant"):
        getattr(exported, kernel.name)(
            torch.empty((48, 40), dtype=torch.bfloat16), args[1], _launcher=hold
        )
    hold.assert_not_called()


def _size_key(x: torch.Tensor) -> tuple[int, torch.dtype]:
    return (x.numel(), x.dtype)


@skipUnlessBackends(["cute"])
def test_dynamic_user_key_snapshot_and_named_argument_dispatch(tmp_path: Path) -> None:
    function = _function(tmp_path)
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function,
        backend="cute",
        static_shapes=False,
        key=_size_key,
        cute_structural_policy=policy,
    )
    args = (torch.empty(37),)
    bound = kernel._bind_isolated(args)
    configs = [helion.Config(block_sizes=[32]), helion.Config(block_sizes=[64])]
    raw = "def key__pointwise(*args):\n    assert args[1] == 4\n    return int(args[0] > 64)\ndef autotune__pointwise(*args):\n    return {}\n"
    model = tmp_path / "model.py"
    model.write_text(attach_model_policy(raw, {kernel.name: configs}, policy))
    cache = AOTAutotuneCache(BaseSearch(bound, args))
    with patch.object(cache, "_find_heuristic_file", return_value=model):
        cache._maybe_run_compile()
    path = policy_path(tmp_path / "original__pointwise_standalone.py", policy)
    module = _module(path)
    for size, index in ((37, 0), (101, 1)):
        x = torch.empty(size)
        calls = []
        module._pointwise(
            x=x, _launcher=lambda *args, calls=calls, **kwargs: calls.append(args[0])
        )
        assert calls[0] is module._helion_aot_modules[index]._helion__pointwise


def test_bad_model_and_unrelocatable_user_key_do_not_replace_export(
    tmp_path: Path,
) -> None:
    policy = CuteStructuralPolicy()
    model = attach_model_policy(
        "def key_demo(*args):\n    return 0\ndef autotune_demo(*args):\n    return {}\n",
        {"demo": [helion.Config()]},
        policy,
    )
    path = policy_path(tmp_path / "demo_standalone.py", policy)
    path.write_text("prior artifact")
    for code, key in (
        (model.replace("'version': 1", "'version': 2"), None),
        (model, lambda x: x.shape),
    ):
        with pytest.raises(StructuralPolicyError):
            generate_standalone_file(
                "demo",
                ["def demo(x):\n    return x\n"],
                code,
                tmp_path,
                structural_policy=policy,
                user_key=key,
            )
        assert path.read_text() == "prior artifact"


@pytest.mark.parametrize("setting", CUTE_STRUCTURAL_SETTINGS)
@skipUnlessBackends(["cute"])
def test_static_export_accumulates_selected_configs_and_own_constants(
    setting: str, tmp_path: Path
) -> None:
    original, first, _marker = _case(setting)
    function = _relocate(original, tmp_path)
    policy = CuteStructuralPolicy(**{setting: True})
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=True, cute_structural_policy=policy
    )
    second = (
        torch.empty((first[0].shape[0] * 2, *first[0].shape[1:]), dtype=first[0].dtype),
        *first[1:],
    )
    config = kernel._bind_isolated(first).config_spec.default_config()
    configs = [config, copy.deepcopy(config)]
    configs[1].block_sizes[0] *= 2
    model = tmp_path / "model.py"
    model.write_text(_model(kernel.name, configs, policy, first[0].shape[0]))
    generated = []
    for index, args in enumerate((first, second)):
        bound = kernel._bind_isolated(args)
        cache = AOTAutotuneCache(BaseSearch(bound, args))
        codegen = bound.to_triton_code

        def capture(
            config: helion.Config, generate: Callable[..., str] = codegen
        ) -> str:
            code = generate(config)
            generated.append(code)
            return code

        with (
            patch.object(cache, "_find_heuristic_file", return_value=model),
            patch.object(bound, "to_triton_code", side_effect=capture) as compile_code,
        ):
            cache._maybe_run_compile()
        compile_code.assert_called_once_with(configs[index])
    path = policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    module = _module(path)
    assert len(module._helion_aot_modules) == 2
    assert set(module._helion_aot_sources) == set(generated)
    for args, code in zip((first, second), generated, strict=True):
        _, actual = _call(getattr(module, kernel.name), args)
        _, expected = _call(getattr(PyCodeCache.load(code), kernel.name), args)
        assert [
            {k: v for k, v in call.items() if k != "kernel"} for call in actual
        ] == [{k: v for k, v in call.items() if k != "kernel"} for call in expected]


def _two_inputs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + y[tile]
    return out


def _constant_scale(x: torch.Tensor, factor: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@pytest.mark.parametrize("static", [False, True])
@skipUnlessBackends(["cute"])
def test_export_guards_tensor_identity_and_accepts_fresh_pointers(
    static: bool, tmp_path: Path
) -> None:
    function = _relocate(_two_inputs, tmp_path)
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=static, cute_structural_policy=policy
    )
    args = (torch.empty(37), torch.empty(37))
    bound = kernel._bind_isolated(args)
    config = bound.config_spec.default_config()
    raw = f"def key_{kernel.name}(*args):\n    return 0\ndef autotune_{kernel.name}(*args):\n    return {{}}\n"
    model = tmp_path / "model.py"
    model.write_text(attach_model_policy(raw, {kernel.name: [config]}, policy))
    cache = AOTAutotuneCache(BaseSearch(bound, args))
    with patch.object(cache, "_find_heuristic_file", return_value=model):
        cache._maybe_run_compile()
    module = _module(
        policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    )
    wrapper = getattr(module, kernel.name)
    _call(wrapper, (torch.empty(37), torch.empty(37)))
    hold = Mock()
    with pytest.raises(ValueError, match="No standalone variant"):
        wrapper(args[0], args[0], _launcher=hold)
    hold.assert_not_called()
    misaligned = torch.empty(38)[1:]
    overlap = torch.empty(37)
    for changed in ((misaligned, torch.empty(37)), (overlap[:], overlap[:])):
        with pytest.raises(ValueError, match="No standalone variant"):
            wrapper(*changed, _launcher=hold)
        hold.assert_not_called()
    if not static:
        # Same registered storage/alignment class remains truly dynamic.
        for count in (1, 53, 101):
            fresh = tuple(torch.empty(count) for _ in range(2))
            result, calls = _call(wrapper, fresh)
            assert result.shape == (count,)
            assert len(calls) == 1
        # Other scalar/tail/stride classes need the same rebind as BoundKernel.
        # Exporting those observations accumulates their generated variants.
        extra = [
            tuple(torch.empty(count * 3)[::3] for _ in range(2)) for count in (0, 2, 91)
        ]
        for fresh in extra:
            current = kernel._bind_isolated(fresh)
            current_cache = AOTAutotuneCache(BaseSearch(current, fresh))
            with patch.object(
                current_cache, "_find_heuristic_file", return_value=model
            ):
                current_cache._maybe_run_compile()
        module = _module(
            policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
        )
        wrapper = getattr(module, kernel.name)
        for fresh in extra:
            result, calls = _call(wrapper, fresh)
            assert result.shape == fresh[0].shape
            assert len(calls) == 1
        # Metadata-only large view: the automatic index width would change.
        large = torch.empty(1).expand(2**31)
        with pytest.raises(ValueError, match="No standalone variant"):
            wrapper(large, torch.empty(1).expand(2**31), _launcher=hold)
        hold.assert_not_called()


@skipUnlessBackends(["cute"])
def test_dynamic_constexpr_export_requires_the_observed_bound_signature(
    tmp_path: Path,
) -> None:
    function = _relocate(_constant_scale, tmp_path)
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=False, cute_structural_policy=policy
    )
    args = (torch.empty(37), 2)
    bound = kernel._bind_isolated(args)
    config = bound.config_spec.default_config()
    model = tmp_path / "model.py"
    raw = f"def key_{kernel.name}(*args):\n    return 0\ndef autotune_{kernel.name}(*args):\n    return {{}}\n"
    model.write_text(attach_model_policy(raw, {kernel.name: [config]}, policy))
    cache = AOTAutotuneCache(BaseSearch(bound, args))
    with patch.object(cache, "_find_heuristic_file", return_value=model):
        cache._maybe_run_compile()
    module = _module(
        policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    )
    wrapper = getattr(module, kernel.name)
    _call(wrapper, (torch.empty(37), 2))
    hold = Mock()
    with pytest.raises(ValueError, match="No standalone variant"):
        wrapper(torch.empty(37), 3, _launcher=hold)
    hold.assert_not_called()


@skipUnlessBackends(["cute"])
def test_compile_deduplication_includes_original_source_bytes(tmp_path: Path) -> None:
    function = _function(tmp_path)
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=False, cute_structural_policy=policy
    )
    args = (torch.empty(37),)
    bound = kernel._bind_isolated(args)
    model = tmp_path / "model.py"
    config = bound.config_spec.default_config()
    raw = f"def key_{kernel.name}(*args):\n    return 0\ndef autotune_{kernel.name}(*args):\n    return {{}}\n"
    model.write_text(attach_model_policy(raw, {kernel.name: [config]}, policy))
    cache = AOTAutotuneCache(BaseSearch(bound, args))
    with (
        patch.object(cache, "_find_heuristic_file", return_value=model),
        patch.object(
            bound, "to_triton_code", wraps=bound.to_triton_code
        ) as compile_code,
    ):
        cache._maybe_run_compile()
        cache._maybe_run_compile()
        assert compile_code.call_count == 1
        source = Path(function.__code__.co_filename)
        source.write_text(source.read_text() + "\n# new exact source identity\n")
        cache._maybe_run_compile()
        assert compile_code.call_count == 2


@pytest.mark.parametrize("kind", ["content", "opaque", "projection", "storage_changed"])
@skipUnlessBackends(["cute"])
def test_unrelocatable_bound_guard_cannot_replace_an_existing_export(
    kind: str, tmp_path: Path
) -> None:
    from torch._dynamo.source import AttrSource

    function = _function(tmp_path)
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=False, cute_structural_policy=policy
    )
    args = (torch.empty(37),)
    bound = kernel._bind_isolated(args)
    config = bound.config_spec.default_config()
    model = tmp_path / "model.py"
    raw = f"def key_{kernel.name}(*args):\n    return 0\ndef autotune_{kernel.name}(*args):\n    return {{}}\n"
    model.write_text(attach_model_policy(raw, {kernel.name: [config]}, policy))
    cache = AOTAutotuneCache(BaseSearch(bound, args))
    with patch.object(cache, "_find_heuristic_file", return_value=model):
        cache._maybe_run_compile()
    path = policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    previous = path.read_bytes()
    key, specialization = next(iter(bound.env.runtime_input_specializations.items()))
    if kind == "content":
        invalid = dataclasses.replace(
            specialization, reusable_tensor_properties=frozenset()
        )
    elif kind == "opaque":
        invalid = dataclasses.replace(specialization, classifier=lambda values: None)
    elif kind == "projection":
        invalid = dataclasses.replace(
            specialization, sources=(AttrSource(specialization.sources[0], "data"),)
        )
    else:
        args[0].set_(torch.empty(38)[1:])
        invalid = specialization
    bound.env.runtime_input_specializations[key] = invalid
    with (
        patch.object(cache, "_find_heuristic_file", return_value=model),
        pytest.raises(StructuralPolicyError),
    ):
        cache._maybe_run_compile()
    assert path.read_bytes() == previous


@skipUnlessBackends(["cute"])
def test_export_classifier_identity_and_input_projection_match_bound_kernel(
    tmp_path: Path,
) -> None:
    function = _relocate(_two_inputs, tmp_path)
    kernel = helion.aot_kernel(
        function,
        backend="cute",
        static_shapes=False,
        cute_structural_policy=CuteStructuralPolicy(),
    )
    args = (torch.empty(37), torch.empty(37))
    bound = kernel._bind_isolated(args)
    guards, observed = export_runtime_guards(bound, args)
    from helion.autotuner.aot_compile import _standalone_value_key

    expected = tuple(
        (key, _standalone_value_key(specialization.classifier(args)))
        for key, specialization in sorted(
            bound.env.runtime_input_specializations.items()
        )
    )
    assert observed == expected
    assert {guard[5] for guard in guards} == {("data_ptr",), ("storage_span",)}
    broken = list(guards)
    fields = list(broken[0])
    fields[3] = "0" * 64
    broken[0] = tuple(fields)
    with pytest.raises(ValueError, match="source identity changed"):
        _export_classifiers(tuple(broken))
    key, specialization = next(iter(bound.env.runtime_input_specializations.items()))
    original = specialization.classifier
    cloned = types.FunctionType(
        original.__code__, original.__globals__, original.__name__
    )
    cloned.__qualname__ = original.__qualname__
    bound.env.runtime_input_specializations[key] = dataclasses.replace(
        specialization, classifier=cloned
    )
    assert export_runtime_guards(bound, args) == (guards, observed)
    changed_globals = types.FunctionType(
        original.__code__, original.__globals__.copy(), original.__name__
    )
    changed_globals.__qualname__ = original.__qualname__
    bound.env.runtime_input_specializations[key] = dataclasses.replace(
        specialization, classifier=changed_globals
    )
    with pytest.raises(StructuralPolicyError, match="declared global"):
        export_runtime_guards(bound, args)
