from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Hashable
from typing import cast
import unittest

from torch._dynamo.source import LocalSource

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import RuntimeInputSpecialization
from helion.runtime.kernel import BoundKernel

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class _PrefixClassifier:
    prefix: str

    def __call__(self, values: Sequence[object]) -> Hashable:
        return self.prefix, tuple(values)


class TestRuntimeInputSpecialization(unittest.TestCase):
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
