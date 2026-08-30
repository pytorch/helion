from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Hashable
from typing import cast
import unittest
from unittest.mock import patch

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
from helion._compiler.device_ir import _finalize_cute_tcgen05_search_planning
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion.language.matmul_ops import _plan_cute_tcgen05_search_candidate
from helion.runtime.cute.launcher import _Tcgen05GroupedWorklistCompatibilityClassifier
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import _input_tensor_aliases

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
