from __future__ import annotations

import copy
import dataclasses
import functools
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
from torch._dynamo.source import GetItemSource
from torch._inductor.codecache import PyCodeCache
from torch.utils._pytree import tree_leaves

import helion
from helion._testing.cute_structural import _case
from helion._testing.cute_structural import _cpu_only  # noqa: F401
from helion._testing.cute_structural import _function
from helion._testing.cute_structural import _module
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_compile import generate_standalone_file
from helion.autotuner.aot_structural_export import _export_call_key
from helion.autotuner.aot_structural_export import _export_classifiers
from helion.autotuner.aot_structural_export import _export_guard_key
from helion.autotuner.aot_structural_export import export_runtime_guards
from helion.autotuner.aot_structural_policy import attach_model_policy
from helion.autotuner.aot_structural_policy import policy_path
from helion.autotuner.base_search import BaseSearch
import helion.language as hl
from helion.runtime.cute.structural_config import StructuralPolicyError
from helion.runtime.cute.structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute.structural_policy import CuteStructuralPolicy

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from typing import Any


@pytest.fixture(autouse=True)
def _shared_budget() -> Iterator[None]:
    with patch(
        "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
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


def _two_inputs(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # Writing an external output registers the cache-specialized storage
    # alias facts; a fresh host allocation would need no such guard.
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + y[tile]
    return out


def _constant_scale(x: torch.Tensor, factor: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@pytest.mark.parametrize("static", [False, True])
def test_export_guards_tensor_identity_and_accepts_fresh_pointers(
    static: bool, tmp_path: Path
) -> None:
    function = _relocate(_two_inputs, tmp_path)
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=static, cute_structural_policy=policy
    )
    args = (torch.empty(37), torch.empty(37), torch.empty(37))
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
    _call(wrapper, (torch.empty(37), torch.empty(37), torch.empty(37)))
    hold = Mock()
    with pytest.raises(ValueError, match="No standalone variant"):
        wrapper(args[0], args[0], args[2], _launcher=hold)
    hold.assert_not_called()
    misaligned = torch.empty(38)[1:]
    overlap = torch.empty(37)
    for changed in (
        (misaligned, torch.empty(37), torch.empty(37)),
        (overlap[:], overlap[:], torch.empty(37)),
        (torch.empty(37), overlap[:], overlap[:]),
    ):
        with pytest.raises(ValueError, match="No standalone variant"):
            wrapper(*changed, _launcher=hold)
        hold.assert_not_called()
    if not static:
        # FP32 extents keep the same remainder in the 32-byte alignment class.
        for count in (5, 53, 101):
            fresh = tuple(torch.empty(count) for _ in range(3))
            result, calls = _call(wrapper, fresh)
            assert result.shape == (count,)
            assert len(calls) == 1
        singleton = tuple(torch.empty(1) for _ in range(3))
        with pytest.raises(ValueError, match="No standalone variant"):
            wrapper(*singleton, _launcher=hold)
        hold.assert_not_called()
        # Other scalar/tail/stride classes need the same rebind as BoundKernel.
        # Exporting those observations accumulates their generated variants.
        extra = [
            singleton,
            *(
                tuple(torch.empty(count * 3)[::3] for _ in range(3))
                for count in (0, 2, 91)
            ),
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
            wrapper(
                large,
                torch.empty(1).expand(2**31),
                torch.empty(1).expand(2**31),
                _launcher=hold,
            )
        hold.assert_not_called()


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
def test_unrelocatable_bound_guard_cannot_replace_an_existing_export(
    kind: str, tmp_path: Path
) -> None:
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
    key, specialization = next(
        (key, value)
        for key, value in bound.env.runtime_input_specializations.items()
        if value.reusable_tensor_properties
    )
    if kind == "content":
        invalid = dataclasses.replace(
            specialization, reusable_tensor_properties=frozenset()
        )
    elif kind == "opaque":
        invalid = dataclasses.replace(specialization, classifier=lambda values: None)
    elif kind == "projection":
        invalid = dataclasses.replace(
            specialization,
            sources=(GetItemSource(specialization.sources[0], slice(None)),),
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


def _canonical_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i, j in hl.tile(x.shape):
        out[i, j] = x[i, j] + 1
    return out


def _stride_extent_copy(x: torch.Tensor) -> torch.Tensor:
    intermediate = torch.empty((x.stride(0), 2), device=x.device, dtype=x.dtype)
    out = torch.empty_like(intermediate)
    for i, j in hl.tile(out.shape):
        out[i, j] = 1
    return out


def _optional_output_copy(
    x: torch.Tensor, out: torch.Tensor, allocate: hl.constexpr
) -> torch.Tensor:
    if allocate:
        result = torch.empty_like(x)
    else:
        result = out
    for i, j in hl.tile(x.shape):
        result[i, j] = x[i, j] + 1
    return result


@functools.cache
def _legacy_export_case() -> tuple[Any, ...]:
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        _optional_output_copy,
        backend="cute",
        static_shapes=True,
        cute_structural_policy=policy,
    )
    args = (torch.empty(4, 8), torch.empty(4, 8), False)
    bound = kernel._bind_isolated(args)
    config = bound.config_spec.default_config()
    code = bound.to_triton_code(config)
    guards, observed = export_runtime_guards(bound, args)
    assert guards and all(len(guard) == 7 for guard in guards)
    return policy, kernel.name, args, config, code, guards, observed


@pytest.mark.parametrize("mode", ["static", "dynamic", "model"])
def test_legacy_guard_export_dispatch_and_no_match_diagnostic(
    mode: str, tmp_path: Path
) -> None:
    policy, name, args, config, code, guards, observed = _legacy_export_case()
    options: dict[str, Any] = {}
    if mode == "static":
        options["dispatch_keys"] = [(_export_call_key(args), observed)]
    elif mode == "dynamic":
        options["dynamic_groups"] = [
            (("dynamic", (_export_call_key(args, dynamic=True), observed)), [0])
        ]
    path = generate_standalone_file(
        name,
        [code],
        _model(name, [config], policy, 100),
        tmp_path,
        structural_policy=policy,
        runtime_guards=guards,
        **options,
    )
    exported = _module(path)
    wrapper = getattr(exported, name)
    expected_result, expected_calls = _call(getattr(PyCodeCache.load(code), name), args)
    result, calls = _call(wrapper, args)
    assert _metadata(result) == _metadata(expected_result)
    assert [{k: v for k, v in call.items() if k != "kernel"} for call in calls] == [
        {k: v for k, v in call.items() if k != "kernel"} for call in expected_calls
    ]
    assert exported._helion_aot_sources == (code,)
    if mode != "model":
        incompatible = (torch.empty(8), torch.empty(8), False)
        launcher = Mock()
        with pytest.raises(ValueError, match="No standalone variant") as error:
            wrapper(*incompatible, _launcher=launcher)
        assert repr(_export_call_key(incompatible)) in str(error.value)
        launcher.assert_not_called()


def test_conflicting_guard_arguments_do_not_replace_export(tmp_path: Path) -> None:
    policy = CuteStructuralPolicy()
    guard = ("dummy", "module", "classifier", "hash", "fact", (), ())
    path = policy_path(tmp_path / "demo_standalone.py", policy)
    path.write_text("prior artifact")
    with pytest.raises(
        StructuralPolicyError, match="runtime_guards.*runtime_guard_plans"
    ):
        generate_standalone_file(
            "demo",
            ["def demo(x):\n    return x\n"],
            _model("demo", [helion.Config()], policy, 100),
            tmp_path,
            dispatch_keys=[((), (), ())],
            structural_policy=policy,
            runtime_guards=(guard,),
            runtime_guard_plans=((),),
        )
    assert path.read_text() == "prior artifact"


@pytest.mark.parametrize("dynamic", [False, True])
def test_unlisted_guard_plan_does_not_replace_export(
    dynamic: bool, tmp_path: Path
) -> None:
    policy = CuteStructuralPolicy()
    unlisted = (("dummy", "module", "classifier", "hash", "fact", (), ()),)
    keys = [((), (), ()), (unlisted, (), ())]
    options: dict[str, Any] = {}
    if dynamic:
        options["dynamic_groups"] = [
            (("dynamic", key), [index]) for index, key in enumerate(keys)
        ]
    else:
        options["dispatch_keys"] = keys
    path = policy_path(tmp_path / "demo_standalone.py", policy)
    path.write_text("prior artifact")
    with pytest.raises(StructuralPolicyError, match="plan.*runtime_guard_plans"):
        generate_standalone_file(
            "demo",
            ["def demo(x):\n    return x\n"] * 2,
            _model("demo", [helion.Config()], policy, 100),
            tmp_path,
            structural_policy=policy,
            runtime_guard_plans=((),),
            **options,
        )
    assert path.read_text() == "prior artifact"


def test_guard_only_no_match_reports_observed_plan_keys(tmp_path: Path) -> None:
    policy, name, args, config, code, guards, observed = _legacy_export_case()
    path = generate_standalone_file(
        name,
        [code],
        _model(name, [config], policy, 100),
        tmp_path,
        dispatch_keys=[(guards, _export_call_key(args), observed)],
        structural_policy=policy,
        runtime_guard_plans=(guards,),
    )
    wrapper = getattr(_module(path), name)
    _call(wrapper, args)
    alias_args = (args[0], args[0].view_as(args[0]), False)
    assert _export_call_key(alias_args) == _export_call_key(args)
    changed_guards = _export_guard_key(alias_args, guards, _export_classifiers(guards))
    assert changed_guards != observed
    launcher = Mock()
    with pytest.raises(ValueError, match="No standalone variant") as error:
        wrapper(*alias_args, _launcher=launcher)
    assert repr(changed_guards) in str(error.value)
    launcher.assert_not_called()


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("canonical_first", [False, True])
def test_export_retains_different_layout_guard_plans_in_both_orders(
    static: bool, canonical_first: bool, tmp_path: Path
) -> None:
    function = _relocate(_optional_output_copy if static else _canonical_copy, tmp_path)
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=static, cute_structural_policy=policy
    )
    inputs = [(torch.empty(4, 8),), (torch.empty_strided((4, 8), (16, 1)),)]
    if static:
        # Static dimensions need no canonical classifier. An external output
        # introduces a distinct storage plan through the same public exporter.
        inputs = [
            (torch.empty(4, 8), torch.empty(4, 8), True),
            (torch.empty(4, 8), torch.empty(4, 8), False),
        ]
    if not canonical_first:
        inputs.reverse()
    bound = kernel._bind_isolated(inputs[0])
    model = tmp_path / "model.py"
    model.write_text(
        _model(kernel.name, [bound.config_spec.default_config()], policy, 100)
    )
    generated = []
    plans = []
    for args in inputs:
        current = kernel._bind_isolated(args)
        cache = AOTAutotuneCache(BaseSearch(current, args))
        original_codegen = current.to_triton_code

        def capture(
            config: helion.Config, generate: Callable[..., str] = original_codegen
        ) -> str:
            code = generate(config)
            generated.append(code)
            return code

        with (
            patch.object(cache, "_find_heuristic_file", return_value=model),
            patch.object(current, "to_triton_code", side_effect=capture),
        ):
            cache._maybe_run_compile()
        plans.append(export_runtime_guards(current, args)[0])
    assert len(generated) == 2 and plans[0] != plans[1]
    path = policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    exported = _module(path)
    assert sorted(exported._helion_aot_sources) == sorted(generated)
    utility_source = Path(exported._helion_aot_utilities.__file__).read_text()
    assert (
        "from helion._compiler.runtime_tensor_guards import _CanonicalLayoutClassifier"
        in utility_source
    )
    for args, code in zip(inputs, generated, strict=True):
        direct = getattr(PyCodeCache.load(code), kernel.name)
        expected_result, expected_calls = _call(direct, args)
        actual_result, actual_calls = _call(getattr(exported, kernel.name), args)
        assert _metadata(actual_result) == _metadata(expected_result)
        assert len(actual_calls) == len(expected_calls) == 1
        assert {k: v for k, v in actual_calls[0].items() if k != "kernel"} == {
            k: v for k, v in expected_calls[0].items() if k != "kernel"
        }


@pytest.mark.parametrize("stride_extent", [False, True])
def test_export_rank_mismatch_refuses_before_launch(
    stride_extent: bool, tmp_path: Path
) -> None:
    function = _relocate(
        _stride_extent_copy if stride_extent else _canonical_copy, tmp_path
    )
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=False, cute_structural_policy=policy
    )
    args = (
        (torch.empty_strided((3, 2), (5, 1)),)
        if stride_extent
        else (torch.empty(4, 6),)
    )
    bound = kernel._bind_isolated(args)
    model = tmp_path / "model.py"
    model.write_text(
        _model(kernel.name, [bound.config_spec.default_config()], policy, 100)
    )
    cache = AOTAutotuneCache(BaseSearch(bound, args))
    with patch.object(cache, "_find_heuristic_file", return_value=model):
        cache._maybe_run_compile()
    path = policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    wrapper = getattr(_module(path), kernel.name)
    _call(wrapper, args)
    incompatible = torch.empty(()) if stride_extent else torch.empty(6)
    launcher = Mock()
    with pytest.raises(ValueError, match="No standalone variant"):
        wrapper(incompatible, _launcher=launcher)
    launcher.assert_not_called()


@pytest.mark.parametrize("stride_extent", [False, True])
def test_canonical_metadata_export_replays_live_layout_and_extent_classes(
    stride_extent: bool, tmp_path: Path
) -> None:
    function = _relocate(
        _stride_extent_copy if stride_extent else _canonical_copy, tmp_path
    )
    policy = CuteStructuralPolicy()
    kernel = helion.aot_kernel(
        function, backend="cute", static_shapes=False, cute_structural_policy=policy
    )
    args = (
        (torch.empty_strided((3, 2), (5, 1)),)
        if stride_extent
        else (torch.empty(4, 6),)
    )
    bound = kernel._bind_isolated(args)
    guards, observed = export_runtime_guards(bound, args)
    metadata_guards = [guard for guard in guards if len(guard) == 8]
    assert metadata_guards and all(guard[5] == () for guard in metadata_guards)
    assert any(
        step[0] == ("stride" if stride_extent else "size")
        for guard in metadata_guards
        for _index, path in guard[6]
        for step in path
        if isinstance(step, tuple)
    )
    assert observed
    config = bound.config_spec.default_config()
    model = tmp_path / "model.py"
    model.write_text(_model(kernel.name, [config], policy, 100))
    cache = AOTAutotuneCache(BaseSearch(bound, args))
    with patch.object(cache, "_find_heuristic_file", return_value=model):
        cache._maybe_run_compile()
    path = policy_path(tmp_path / f"original_{kernel.name}_standalone.py", policy)
    wrapper = getattr(_module(path), kernel.name)
    _call(wrapper, args)
    _call(wrapper, (torch.empty_strided(args[0].size(), args[0].stride()),))
    utilities = wrapper.__globals__["_helion_aot_utilities"]
    metadata_plan = tuple(metadata_guards)
    classifiers = utilities._export_classifiers(metadata_plan)
    metadata_key = utilities._export_guard_key(args, metadata_plan, classifiers)
    compatible = (
        torch.empty_strided((3, 2), (7, 1)) if stride_extent else torch.empty(8, 6)
    )
    # The independent vector-alignment guard also records exact shapes/strides.
    # Exercise canonical metadata replay without weakening that existing guard.
    assert (
        utilities._export_guard_key((compatible,), metadata_plan, classifiers)
        == metadata_key
    )
    incompatible = (
        [torch.empty_strided((3, 2), (1, 1))]
        if stride_extent
        else [torch.empty(4, 1), torch.empty(6, 4).t()]
    )
    for value in incompatible:
        assert (
            utilities._export_guard_key((value,), metadata_plan, classifiers)
            != metadata_key
        )
        launcher = Mock()
        with pytest.raises(ValueError, match="No standalone variant"):
            wrapper(value, _launcher=launcher)
        launcher.assert_not_called()
    # Mutating the original argument cannot rewrite the binding snapshot.
    snapshot = bound.env.bound_metadata_specialization_results.copy()
    args[0].set_(incompatible[0])
    with pytest.raises(StructuralPolicyError, match="changed their bound"):
        export_runtime_guards(bound, args)
    assert bound.env.bound_metadata_specialization_results == snapshot


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
    args = (torch.empty(37), torch.empty(37), torch.empty(37))
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
