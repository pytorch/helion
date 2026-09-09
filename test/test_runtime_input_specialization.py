from __future__ import annotations

import contextvars
import dataclasses
import inspect
import operator
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Hashable
from typing import cast
import unittest
from unittest.mock import patch
import weakref

import sympy
import torch
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import RuntimeInputSpecialization
from helion._compiler.compile_environment import _symint_free_symbols
from helion._compiler.cute.memory_ops import (
    _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
)
from helion._compiler.cute.memory_ops import _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
from helion._compiler.cute.memory_ops import _persistent_vec_alignment_matrix_signature
from helion._compiler.cute.memory_ops import _tensor_storage_disjoint_matrix_signature
from helion._compiler.cute.memory_ops import runtime_tensor_has_specialized_alignment
from helion._compiler.cute.memory_ops import runtime_tensors_are_proven_disjoint
from helion._compiler.device_ir import _finalize_cute_tcgen05_search_planning
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion.autotuner.base_cache import BoundKernelInMemoryCacheKey
from helion.language.matmul_ops import _plan_cute_tcgen05_search_candidate
from helion.runtime.cute.launcher import _Tcgen05GroupedWorklistCompatibilityClassifier
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import Kernel
from helion.runtime.kernel import _input_tensor_aliases
from helion.runtime.kernel import _partition_prepared_extra_guards
from helion.runtime.kernel import _PreparedCall
from helion.runtime.kernel import _PreparedMetadataSpecializationExtractor
from helion.runtime.kernel import _RuntimeInputSpecializationExtractor

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class _PrefixClassifier:
    prefix: str

    def __call__(self, values: Sequence[object]) -> Hashable:
        return self.prefix, tuple(values)


@onlyBackends(["cute"])
class TestCuteRuntimeInputSpecialization(unittest.TestCase):
    def test_conflicting_tcgen05_analysis_keys_still_publish_all_guards(self) -> None:
        first = sympy.Symbol("first")
        second = sympy.Symbol("second")
        env = SimpleNamespace(specialized_vars=set())
        planning_results = (
            SimpleNamespace(
                required_specialized_vars=frozenset({first}),
                guards_complete=True,
            ),
            SimpleNamespace(
                required_specialized_vars=frozenset({second}),
                guards_complete=True,
            ),
        )

        self.assertFalse(
            _finalize_cute_tcgen05_search_planning(
                cast("CompileEnvironment", env),
                cast("Any", planning_results),
                {("first",), ("second",)},
            )
        )
        self.assertEqual(env.specialized_vars, {first, second})

        third = sympy.Symbol("third")
        incomplete = SimpleNamespace(
            required_specialized_vars=frozenset({third}),
            guards_complete=False,
        )
        self.assertFalse(
            _finalize_cute_tcgen05_search_planning(
                cast("CompileEnvironment", env),
                cast("Any", (planning_results[0], incomplete)),
                {("shared",)},
            )
        )
        self.assertEqual(env.specialized_vars, {first, second, third})

    def test_dynamic_tcgen05_candidate_plans_are_idempotent_and_specialized(
        self,
    ) -> None:
        shape_env = ShapeEnv()

        def backed_size(name: str, dim: int, hint: int) -> torch.SymInt:
            source = TensorPropertySource(
                LocalSource(name, is_input=True),
                TensorProperty.SIZE,
                dim,
            )
            symbol = shape_env.create_symbol(
                hint,
                source,
                dynamic_dim=DimDynamic.DYNAMIC,
            )
            return cast(
                "torch.SymInt",
                shape_env.create_symintnode(symbol, hint=hint, source=source),
            )

        tile_m = shape_env.create_unbacked_symint()
        tile_n = shape_env.create_unbacked_symint()
        tile_k = shape_env.create_unbacked_symint()
        problem_m = backed_size("lhs", 0, 256)
        problem_n = backed_size("rhs", 1, 512)
        problem_k = backed_size("lhs", 1, 128)
        with FakeTensorMode(shape_env=shape_env):
            lhs = torch.empty((tile_m, tile_k), device=DEVICE, dtype=torch.bfloat16)
            rhs = torch.empty((tile_k, tile_n), device=DEVICE, dtype=torch.bfloat16)
        hints = {str(tile_m): 256, str(tile_n): 512, str(tile_k): 128}
        block_ids = {str(tile_m): 0, str(tile_n): 1, str(tile_k): 2}
        env = SimpleNamespace(
            backend_name="cute",
            settings=SimpleNamespace(static_shapes=False),
            shape_env=shape_env,
            specialized_vars=set(),
            specialized_strides=set(),
            tensor_descriptor_layout_guards={},
            runtime_input_specializations={},
            specialize_expr=lambda expr: expr,
            size_hint=lambda size: hints[str(size)],
            get_block_id=lambda size: block_ids.get(str(size)),
            block_sizes=[
                SimpleNamespace(size=problem_m),
                SimpleNamespace(size=problem_n),
                SimpleNamespace(size=problem_k),
            ],
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.cute.mma_support.get_cute_mma_support",
                return_value=SimpleNamespace(tcgen05_f8=True, tcgen05_f16bf16=True),
            ),
        ):
            planning_results = []
            for m_hint, expected_plan in ((32, False), (256, True)):
                with self.subTest(m_hint=m_hint):
                    hints[str(tile_m)] = m_hint
                    env.specialized_vars.clear()
                    planning_result = _plan_cute_tcgen05_search_candidate(
                        lhs,
                        rhs,
                        has_leading_passthrough=False,
                        allow_dynamic_hints=True,
                    )
                    plan = planning_result.plan

                    self.assertEqual(plan is not None, expected_plan)
                    self.assertTrue(planning_result.guards_complete)
                    self.assertFalse(env.specialized_vars)
                    planning_results.append(planning_result)
                    if plan is not None:
                        # DeviceIR may inspect multiple compatible MMAs before
                        # deciding whether there is one shared analysis. Each
                        # candidate must see the same compile state.
                        repeated_result = _plan_cute_tcgen05_search_candidate(
                            lhs,
                            rhs,
                            has_leading_passthrough=False,
                            allow_dynamic_hints=True,
                        )
                        repeated_plan = repeated_result.plan
                        assert repeated_plan is not None
                        self.assertEqual(
                            (plan.static_m, plan.static_n, plan.static_k),
                            (None, None, None),
                        )
                        self.assertEqual(planning_result, repeated_result)
            self.assertEqual(len(planning_results), 2)
            required_specialized_vars = frozenset().union(
                *(result.required_specialized_vars for result in planning_results)
            )
            self.assertEqual(
                required_specialized_vars,
                frozenset().union(
                    _symint_free_symbols(problem_m),
                    _symint_free_symbols(problem_n),
                    _symint_free_symbols(problem_k),
                ),
            )
            env.specialized_vars.update(required_specialized_vars)
            self.assertEqual(
                env.specialized_vars,
                set().union(
                    _symint_free_symbols(problem_m),
                    _symint_free_symbols(problem_n),
                    _symint_free_symbols(problem_k),
                ),
            )
            bound = cast("BoundKernel[Any]", object.__new__(BoundKernel))
            bound._env = cast("CompileEnvironment", env)
            bound._config = None
            bound.kernel = cast(
                "Any",
                SimpleNamespace(
                    signature=inspect.signature(lambda lhs, rhs: None),
                    settings=SimpleNamespace(
                        autotune_effort="max", force_autotune=False
                    ),
                    configs=[],
                ),
            )
            extractors = bound._specialize_extra()
            real_lhs = torch.empty((256, 128))
            real_rhs = torch.empty((128, 512))
            self.assertEqual(
                sorted(
                    cast("int", extractor((real_lhs, real_rhs)))
                    for extractor in extractors
                ),
                [128, 256, 512],
            )

            env.block_sizes[1].size = None
            env.specialized_vars.clear()
            missing_result = _plan_cute_tcgen05_search_candidate(
                lhs,
                rhs,
                has_leading_passthrough=False,
                allow_dynamic_hints=True,
            )
            self.assertIsNone(missing_result.plan)
            self.assertFalse(missing_result.guards_complete)
            self.assertFalse(env.specialized_vars)

            env.block_sizes[1].size = shape_env.create_unbacked_symint()
            unbacked_result = _plan_cute_tcgen05_search_candidate(
                lhs,
                rhs,
                has_leading_passthrough=False,
                allow_dynamic_hints=True,
            )
            self.assertIsNone(unbacked_result.plan)
            self.assertFalse(unbacked_result.guards_complete)
            self.assertFalse(env.specialized_vars)

    def test_worklist_classifier_invalidates_on_storage_rebind(self) -> None:
        classifier = _Tcgen05GroupedWorklistCompatibilityClassifier(2, 448)
        worklist = torch.tensor(
            [[0, 0, 224, 224], [1, 224, 224, 224]], dtype=torch.int32
        )
        replacement = torch.tensor(
            [[0, 0, 256, 256], [1, 256, 192, 192]], dtype=torch.int32
        )
        version = worklist._version
        original_data_ptr = worklist.data_ptr()

        self.assertEqual(classifier((worklist,)), (32, 224))
        worklist.data = replacement.data

        self.assertEqual(worklist._version, version)
        self.assertNotEqual(worklist.data_ptr(), original_data_ptr)
        self.assertEqual(classifier((worklist,)), (32,))


class TestRuntimeInputSpecialization(unittest.TestCase):
    def test_set_config_rejects_post_bind_realignment(self) -> None:
        source = LocalSource("value", is_input=True)
        specialization = RuntimeInputSpecialization(
            sources=(source,),
            classifier_identity="alignment_test",
            classifier=_persistent_vec_alignment_matrix_signature,
            reusable_tensor_properties=frozenset(("data_ptr",)),
        )
        env = cast("CompileEnvironment", object.__new__(CompileEnvironment))
        env.runtime_input_specializations = {
            _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY: specialization
        }
        env.bound_runtime_input_specialization_results = {}
        env._runtime_arg_values_by_name = contextvars.ContextVar(
            "test_runtime_arg_values", default=None
        )
        env.tensor_input_source = lambda _tensor: source  # type: ignore[method-assign]

        backing = torch.empty(17, dtype=torch.bfloat16)
        value = backing[1:17]
        aligned = backing[:16]
        self.assertEqual(value.data_ptr() % 4, 2)
        env.snapshot_runtime_input_specialization_results({"value": value})
        value.data = aligned.data
        self.assertEqual(value.data_ptr() % 4, 0)

        bound = cast("Any", object.__new__(BoundKernel))
        bound._env = env
        bound._runtime_tensor_refs_by_name = {"value": weakref.ref(value)}
        bound._normalize_config = lambda config: config
        bound.format_kernel_decorator = lambda _config, _settings: "test"
        bound.kernel = SimpleNamespace(settings=SimpleNamespace())
        observed: list[bool] = []

        def compile_config(_config: object) -> object:
            with bound._runtime_arg_values_for_codegen():
                observed.append(
                    runtime_tensor_has_specialized_alignment(
                        env, cast("Any", object()), 4
                    )
                )
            return lambda: None

        bound.compile_config = compile_config
        BoundKernel.set_config(bound, cast("Any", object()))

        self.assertEqual(observed, [False])

    def test_bound_runtime_specialization_snapshot_is_immutable(self) -> None:
        source = LocalSource("value", is_input=True)
        content_calls = 0

        def content_classifier(values: Sequence[object]) -> int:
            nonlocal content_calls
            content_calls += 1
            return int(cast("torch.Tensor", values[0])[0])

        specialization = RuntimeInputSpecialization(
            sources=(source,),
            classifier_identity="pointer_test",
            classifier=lambda values: cast("torch.Tensor", values[0]).data_ptr(),
            reusable_tensor_properties=frozenset(("data_ptr",)),
        )
        env = cast("CompileEnvironment", object.__new__(CompileEnvironment))
        env.runtime_input_specializations = {
            "content": RuntimeInputSpecialization(
                sources=(source,),
                classifier_identity="content_test",
                classifier=content_classifier,
            ),
            "pointer": specialization,
        }
        env.bound_runtime_input_specialization_results = {}
        original = torch.empty(8)
        original_pointer = original.data_ptr()

        env.snapshot_runtime_input_specialization_results({"value": original})
        self.assertEqual(content_calls, 0)
        self.assertNotIn("content", env.bound_runtime_input_specialization_results)
        self.assertFalse(env.runtime_input_specialization_matches_bound("late", None))
        original.data = torch.empty_like(original).data
        self.assertNotEqual(original.data_ptr(), original_pointer)

        self.assertTrue(
            env.runtime_input_specialization_matches_bound("pointer", original_pointer)
        )
        self.assertFalse(
            env.runtime_input_specialization_matches_bound(
                "pointer", original.data_ptr()
            )
        )

    def test_late_rekey_retires_bound_when_runtime_snapshot_changes(self) -> None:
        source = LocalSource("value", is_input=True)
        key = _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
        specialization = RuntimeInputSpecialization(
            sources=(source,),
            classifier_identity="alignment_test",
            classifier=_persistent_vec_alignment_matrix_signature,
            reusable_tensor_properties=frozenset(("data_ptr",)),
        )
        env = cast("CompileEnvironment", object.__new__(CompileEnvironment))
        env.runtime_input_specializations = {key: specialization}
        env.bound_runtime_input_specialization_results = {}
        env._runtime_arg_values_by_name = contextvars.ContextVar(
            "test_late_rekey_runtime_args", default=None
        )
        env.tensor_input_source = lambda _tensor: source  # type: ignore[method-assign]

        backing = torch.empty(17, dtype=torch.bfloat16)
        aligned = backing[:16]
        unaligned = backing[1:17]
        value = backing[:16]
        self.assertEqual(aligned.data_ptr() % 4, 0)

        extractor = _RuntimeInputSpecializationExtractor(
            (operator.itemgetter(0),),
            specialization.classifier,
            frozenset(("data_ptr",)),
            key,
        )
        facts_a = extractor((value,))
        env.bound_runtime_input_specialization_results = {key: facts_a}

        signature = ("signature",)
        bound = cast("Any", object.__new__(BoundKernel))
        bound._reset_generation = 0
        bound._base_spec_key = signature
        bound._env = env
        bound._cache_managed = True
        stale_calls = 0

        def compiled_a(_value: torch.Tensor) -> str:
            nonlocal stale_calls
            stale_calls += 1
            return "stale-a"

        # Model a config compiled and selected while the bound still had facts A.
        config_a = object()
        bound._run = compiled_a
        bound._config = config_a
        bound._compile_cache = {config_a: compiled_a}
        bound._cache_path_map = {config_a: "compiled-a.py"}
        bound._direct_prepared_call = object()
        kernel = cast("Any", object.__new__(Kernel))
        kernel._bind_lock = threading.RLock()
        kernel._specialize_extra_lock = threading.Lock()
        kernel._reset_generation = 0
        kernel._specialization_generation = 0
        kernel._specialization_aliases = {}
        kernel._dispatch_cache = {}
        kernel._prepared_call = None
        kernel._specialize_extra = {signature: [extractor]}
        kernel._compiler_seed_specialize_extra = {signature: ()}
        kernel._has_specialization_extras = True
        kernel._bound_kernels = {
            BoundKernelInMemoryCacheKey(signature, (facts_a,)): bound
        }
        bound.kernel = kernel

        self.assertTrue(
            Kernel._extend_bound_kernel_specializations(
                kernel,
                bound,
                signature,
                [lambda _args: "tail-a"],
                (value,),
            )
        )
        (matching_cache_key,) = kernel._bound_kernels
        self.assertEqual(matching_cache_key.extra_results, (facts_a, "tail-a"))
        self.assertEqual(env.bound_runtime_input_specialization_results[key], facts_a)
        self.assertIs(bound._run, compiled_a)
        self.assertEqual(bound._compile_cache, {config_a: compiled_a})

        value.data = unaligned.data
        facts_b = extractor((value,))
        self.assertNotEqual(facts_a, facts_b)
        self.assertTrue(
            Kernel._extend_bound_kernel_specializations(
                kernel,
                bound,
                signature,
                [lambda _args: "tail-b"],
                (value,),
            )
        )
        self.assertFalse(kernel._bound_kernels)
        self.assertEqual(env.bound_runtime_input_specialization_results[key], facts_a)
        self.assertNotEqual(bound._reset_generation, kernel._reset_generation)
        self.assertIsNone(bound._run)
        self.assertIsNone(bound._config)
        self.assertFalse(bound._compile_cache)
        self.assertFalse(bound._cache_path_map)
        self.assertIsNone(bound._direct_prepared_call)

        value.data = aligned.data
        with env.use_runtime_arg_values({"value": value}):
            self.assertTrue(
                runtime_tensor_has_specialized_alignment(
                    env,
                    cast("Any", object()),
                    4,
                )
            )

        # Even if stale compiled state is assigned again, the retired bound
        # redirects before it can execute that program.
        bound._run = compiled_a
        replacement_calls = 0

        def replacement(_value: torch.Tensor) -> str:
            nonlocal replacement_calls
            replacement_calls += 1
            return "replacement-b"

        kernel.bind = lambda _args: replacement
        value.data = unaligned.data
        self.assertEqual(bound(value), "replacement-b")
        self.assertEqual(replacement_calls, 1)
        self.assertEqual(stale_calls, 0)

    def test_disjoint_fact_rejects_post_bind_alias_change(self) -> None:
        state_source = LocalSource("state", is_input=True)
        other_source = LocalSource("other", is_input=True)
        sources = (state_source, other_source)
        specialization = RuntimeInputSpecialization(
            sources=sources,
            classifier_identity="state_ring_alias_test",
            classifier=_tensor_storage_disjoint_matrix_signature,
            reusable_tensor_properties=frozenset(("storage_span",)),
        )
        backing = torch.empty(64)
        bound_values = (backing[:32], backing[16:48])
        current_values = (torch.empty(32), torch.empty(32))
        bound_result = specialization.classifier(bound_values)
        fake_state = object()
        fake_other = object()
        env = SimpleNamespace(
            input_sources={},
            runtime_arg_values_by_name={
                "state": current_values[0],
                "other": current_values[1],
            },
            runtime_input_specializations={
                _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY: specialization
            },
            tensor_input_source=lambda tensor: (
                state_source if tensor is fake_state else other_source
            ),
            runtime_input_specialization_matches_bound=(
                lambda key, result: (
                    key == _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
                    and result == bound_result
                )
            ),
        )

        with patch(
            "helion._compiler.cute.memory_ops._tensor_alias_sources",
            return_value=sources,
        ):
            self.assertFalse(
                runtime_tensors_are_proven_disjoint(
                    cast("Any", env), cast("Any", fake_state), cast("Any", fake_other)
                )
            )
            current_result = specialization.classifier(current_values)
            env.runtime_input_specialization_matches_bound = lambda key, result: (
                key == _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
                and result == current_result
            )
            self.assertTrue(
                runtime_tensors_are_proven_disjoint(
                    cast("Any", env), cast("Any", fake_state), cast("Any", fake_other)
                )
            )

    def test_state_ring_alias_specialization_distinguishes_storage_views(self) -> None:
        state_source = LocalSource("state", is_input=True)
        other_source = LocalSource("other", is_input=True)
        specialization = RuntimeInputSpecialization(
            sources=(state_source, other_source),
            classifier_identity="state_ring_alias_test",
            classifier=_tensor_storage_disjoint_matrix_signature,
            reusable_tensor_properties=frozenset(("storage_span",)),
        )
        env = SimpleNamespace(
            specialized_vars=set(),
            specialized_strides=set(),
            tensor_descriptor_layout_guards={},
            runtime_input_specializations={"state_ring_alias": specialization},
        )
        kernel = SimpleNamespace(signature=inspect.signature(lambda state, other: None))
        bound = SimpleNamespace(
            _env=env,
            env=env,
            kernel=kernel,
            _fixed_config_for_td_layout_guards=lambda: None,
        )
        extractor = BoundKernel._specialize_extra(cast("BoundKernel[object]", bound))[0]

        state = torch.empty((4, 8))
        other = torch.empty((4, 8))
        backing = torch.empty(64)
        state_view = backing[:32].view(4, 8)
        other_view = backing[16:48].view(4, 8)

        self.assertEqual(state.shape, state_view.shape)
        self.assertEqual(state.stride(), state_view.stride())
        self.assertEqual(extractor((state, other)), (True,))
        self.assertEqual(extractor((state_view, other_view)), (False,))

    def test_direct_stale_bound_rebinds_after_kernel_reset(self) -> None:
        calls = 0

        def replacement(value: object) -> object:
            nonlocal calls
            calls += 1
            return value

        kernel = SimpleNamespace(
            _reset_generation=2,
            bind=lambda _args: replacement,
        )
        bound = cast("Any", object.__new__(BoundKernel))
        bound.kernel = kernel
        bound._cache_managed = True
        bound._reset_generation = 1

        value = object()
        self.assertIs(bound(value), value)
        self.assertEqual(calls, 1)

    def test_direct_bound_call_uses_prepared_fast_path(self) -> None:
        tensor = torch.empty(8)
        kernel = SimpleNamespace(
            _annotations=[torch.Tensor],
            _compute_is_distributed=lambda _args, **_kwargs: False,
            _reset_generation=0,
            _specialization_generation=0,
        )
        bound = cast("Any", object.__new__(BoundKernel))
        bound.kernel = kernel
        bound._cache_managed = True
        bound._reset_generation = 0
        bound._run = lambda value: value
        bound._direct_prepared_call = _PreparedCall(
            cast("Any", bound),
            (tensor,),
            dist_initialized=torch.distributed.is_initialized(),
            extra_guards=(),
            is_distributed=False,
        )

        self.assertIs(bound(tensor), tensor)

    def test_prepared_call_skips_reusable_classifier_until_storage_changes(
        self,
    ) -> None:
        tensor = torch.empty(8)
        calls = 0

        def classifier(values: Sequence[object]) -> Hashable:
            nonlocal calls
            calls += 1
            return cast("torch.Tensor", values[0]).data_ptr()

        extractor = _RuntimeInputSpecializationExtractor(
            (lambda args: cast("Hashable", args[0]),),
            classifier,
            frozenset(("data_ptr",)),
        )
        expected = extractor((tensor,))
        kernel = SimpleNamespace(
            _annotations=[torch.Tensor],
            _compute_is_distributed=lambda _args, **_kwargs: False,
            _reset_generation=0,
            _specialization_generation=0,
        )
        bound = SimpleNamespace(kernel=kernel)
        prepared = _PreparedCall(
            cast("Any", bound),
            (tensor,),
            dist_initialized=torch.distributed.is_initialized(),
            extra_guards=((extractor, expected),),
            is_distributed=False,
        )

        calls = 0
        self.assertTrue(prepared.matches(cast("Any", kernel), (tensor,)))
        self.assertEqual(calls, 0)

        replacement = torch.empty_like(tensor)
        tensor.data = replacement.data
        self.assertFalse(prepared.matches(cast("Any", kernel), (tensor,)))
        self.assertEqual(calls, 1)

        kernel._specialization_generation += 1
        calls = 0
        self.assertFalse(prepared.matches(cast("Any", kernel), (tensor,)))
        self.assertEqual(calls, 0)

    def test_prepared_storage_guard_reuses_only_unchanged_exact_tensors(self) -> None:
        first = torch.empty(8)
        second = torch.empty(8)
        calls = 0

        def classifier(values: Sequence[object]) -> Hashable:
            nonlocal calls
            calls += 1
            return tuple(cast("torch.Tensor", value).data_ptr() for value in values)

        extractor = _RuntimeInputSpecializationExtractor(
            (
                lambda args: cast("Hashable", args[0]),
                lambda args: cast("Hashable", args[1]),
            ),
            classifier,
            frozenset(("data_ptr", "storage_span")),
        )
        expected = extractor((first, second))
        storage_guard, always_check, reusable = _partition_prepared_extra_guards(
            (first, second),
            ((extractor, expected),),
        )
        self.assertIsNotNone(storage_guard)
        assert storage_guard is not None
        self.assertFalse(always_check)
        self.assertEqual(reusable, ((extractor, expected),))
        self.assertTrue(storage_guard((first, second)))

        replacement = torch.empty_like(first)
        original_version = first._version
        first.data = replacement.data
        self.assertEqual(first._version, original_version)
        self.assertFalse(storage_guard((first, second)))
        self.assertEqual(calls, 1)

    def test_prepared_guard_keeps_content_classifiers_and_skips_metadata(self) -> None:
        tensor = torch.arange(4)
        content_calls = 0

        def content_classifier(values: Sequence[object]) -> int:
            nonlocal content_calls
            content_calls += 1
            return int(cast("torch.Tensor", values[0])[0])

        def content_extractor(args: Sequence[object]) -> int:
            return content_classifier((args[0],))

        metadata_extractor = _PreparedMetadataSpecializationExtractor(
            lambda args: cast("torch.Tensor", args[0]).size(0)
        )
        unknown_property_extractor = _RuntimeInputSpecializationExtractor(
            (lambda args: cast("Hashable", args[0]),),
            lambda values: cast("torch.Tensor", values[0]).data_ptr(),
            frozenset(("unknown",)),
        )
        storage_guard, always_check, reusable = _partition_prepared_extra_guards(
            (tensor,),
            (
                (content_extractor, content_extractor((tensor,))),
                (metadata_extractor, metadata_extractor((tensor,))),
                (
                    unknown_property_extractor,
                    unknown_property_extractor((tensor,)),
                ),
            ),
        )
        self.assertIsNone(storage_guard)
        self.assertEqual(
            always_check,
            (
                (content_extractor, 0),
                (unknown_property_extractor, tensor.data_ptr()),
            ),
        )
        self.assertFalse(reusable)

        tensor[0] = 3
        self.assertEqual(always_check[0][0]((tensor,)), 3)
        self.assertEqual(content_calls, 2)

    def test_tensor_alias_key_is_dynamo_safe_for_temporary_views(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            assert _input_tensor_aliases((x.T, y.T)) is None
            return x + y

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        self.assertEqual(_input_tensor_aliases((x, x, y)), (0, 0, 1))
        torch.testing.assert_close(compiled(x, y), x + y)

    def test_extractors_have_deterministic_key_order(self) -> None:
        first = LocalSource("first", is_input=True)
        second = LocalSource("second", is_input=True)
        entries = (
            (
                "z",
                RuntimeInputSpecialization(
                    sources=(second,),
                    classifier_identity="z",
                    classifier=_PrefixClassifier("z"),
                ),
            ),
            (
                "a",
                RuntimeInputSpecialization(
                    sources=(first,),
                    classifier_identity="a",
                    classifier=_PrefixClassifier("a"),
                ),
            ),
        )

        def results(
            ordered_entries: Sequence[tuple[str, RuntimeInputSpecialization]],
        ) -> tuple[Hashable, ...]:
            env = SimpleNamespace(
                specialized_vars=set(),
                specialized_strides=set(),
                tensor_descriptor_layout_guards={},
                runtime_input_specializations=dict(ordered_entries),
            )
            kernel = SimpleNamespace(
                signature=inspect.signature(lambda first, second: None)
            )
            bound = SimpleNamespace(
                _env=env,
                env=env,
                kernel=kernel,
                _fixed_config_for_td_layout_guards=lambda: None,
            )
            extractors = BoundKernel._specialize_extra(
                cast("BoundKernel[object]", bound)
            )
            return tuple(extractor((11, 22)) for extractor in extractors)

        expected = (("a", (11,)), ("z", (22,)))
        self.assertEqual(results(entries), expected)
        self.assertEqual(results(tuple(reversed(entries))), expected)

    def test_equivalent_registration_is_idempotent(self) -> None:
        env = object.__new__(CompileEnvironment)
        env.runtime_input_specializations = {}
        source = LocalSource("value", is_input=True)
        first = RuntimeInputSpecialization(
            sources=(source,),
            classifier_identity="same",
            classifier=_PrefixClassifier("same"),
        )
        equivalent = RuntimeInputSpecialization(
            sources=(source,),
            classifier_identity="same",
            classifier=_PrefixClassifier("same"),
        )

        env.register_runtime_input_specialization("key", first)
        env.register_runtime_input_specialization("key", equivalent)

        self.assertEqual(len(env.runtime_input_specializations), 1)
        self.assertIs(env.runtime_input_specializations["key"], first)

    def test_conflicting_registration_raises(self) -> None:
        env = object.__new__(CompileEnvironment)
        env.runtime_input_specializations = {}
        source = LocalSource("value", is_input=True)
        first = RuntimeInputSpecialization(
            sources=(source,),
            classifier_identity="first",
            classifier=_PrefixClassifier("first"),
        )
        env.register_runtime_input_specialization("key", first)

        conflicts = (
            RuntimeInputSpecialization(
                sources=(source,),
                classifier_identity="different",
                classifier=_PrefixClassifier("different"),
            ),
            RuntimeInputSpecialization(
                sources=(LocalSource("other", is_input=True),),
                classifier_identity="first",
                classifier=_PrefixClassifier("first"),
            ),
        )
        for conflict in conflicts:
            with (
                self.subTest(conflict=conflict),
                self.assertRaisesRegex(
                    RuntimeError,
                    "conflicting runtime input specializations",
                ),
            ):
                env.register_runtime_input_specialization("key", conflict)


if __name__ == "__main__":
    unittest.main()
