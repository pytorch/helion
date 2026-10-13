from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from helion import Config
from helion._compiler.cute.eviction_relevance import PolicyUses
from helion._compiler.cute.eviction_relevance import authority
from helion._compiler.cute.eviction_relevance import collect
from helion._compiler.cute.eviction_relevance import record_load
from helion.autotuner.pattern_search import PatternSearch
from helion.runtime.kernel import BoundKernel


def config(policies=None, **kwargs):
    return Config(
        load_eviction_policies=policies or ["", ""], block_sizes=[16], **kwargs
    )


class TestEvictionRelevance(unittest.TestCase):
    def make_receipt(self, *, second_use=False, unknown=False):
        owner = object()
        with collect(config()) as uses:
            record_load(owner, 0, scalar_bytes=2)
            record_load(owner, 1, scalar_bytes=4)
            if second_use:
                uses.record(
                    owner, 0, ("", "first", "last", "streaming", "l2", "l2f", "l2l")
                )
            if unknown:
                record_load(owner, 0, scalar_bytes=None)
        return uses.finish("authority", "source")

    def test_scalar_classes_preserve_real_l1_choices(self):
        receipt = self.make_receipt()
        for policy in (
            "first",
            "last",
            "streaming",
            "l2_last",
            "l1_l2_first",
            "l1_l2_last",
        ):
            with self.subTest(policy=policy):
                self.assertTrue(receipt.excludes(config([policy, ""])))
                self.assertEqual(
                    receipt.excludes(config(["", policy])),
                    policy in ("l2_last", "l1_l2_first", "l1_l2_last"),
                )

    def test_classes_match_original_scalar_helper(self):
        from helion._compiler.compile_environment import CompileEnvironment
        from helion.language.memory_ops import _cute_scalar_load_expr

        policies = (
            "",
            "first",
            "last",
            "streaming",
            "l2_last",
            "l1_l2_first",
            "l1_l2_last",
        )
        suffixes = (
            "",
            ", level1_eviction_priority='evict_first'",
            ", level1_eviction_priority='evict_last'",
            ", cop='cs'",
            "__l2_last__",
            "__l1_l2_first__",
            "__l1_l2_last__",
        )
        env = SimpleNamespace(
            backend=SimpleNamespace(dtype_str=lambda dtype: str(dtype))
        )
        with patch.object(CompileEnvironment, "current", return_value=env):
            for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.int32):
                owner = object()
                with collect(config([""])) as uses:
                    record_load(owner, 0, scalar_bytes=dtype.itemsize)
                receipt = uses.finish("authority", "source")
                spellings = [
                    _cute_scalar_load_expr("x", [], dtype, eviction_suffix=suffix)
                    for suffix in suffixes
                ]
                for i, policy in enumerate(policies):
                    if policy:
                        self.assertEqual(
                            receipt.excludes(config([policy])),
                            spellings[i] == spellings[0],
                        )

    def test_every_use_must_agree(self):
        self.assertFalse(
            self.make_receipt(second_use=True).excludes(config(["first", ""]))
        )
        self.assertFalse(
            self.make_receipt(unknown=True).excludes(config(["first", ""]))
        )

    def test_unknown_unseen_and_out_of_range_are_conservative(self):
        uses = PolicyUses(config())
        receipt = uses.finish("authority", "source")
        self.assertFalse(receipt.excludes(config(["first", ""])))
        uses.record(object(), 3, None)
        self.assertIsNone(uses.finish("authority", "source"))

    def test_retry_and_multiple_device_functions_disable(self):
        uses = PolicyUses(config())
        uses.record(object(), 0, ("",) * 7)
        uses.record(object(), 0, ("",) * 7)
        self.assertIsNone(uses.finish("authority", "source"))

    def test_changed_other_config_or_multiple_policies_kept(self):
        receipt = self.make_receipt()
        candidate = config(["first", ""])
        candidate.config["block_sizes"] = [32]
        self.assertFalse(receipt.excludes(candidate))
        self.assertFalse(receipt.excludes(config(["first", "l2_last"])))
        self.assertFalse(receipt.excludes(config(["first"])))

    def test_omitted_or_empty_policy_domain_keeps_candidates(self):
        for base in (
            Config(block_sizes=[16]),
            Config(block_sizes=[16], load_eviction_policies=[]),
        ):
            with self.subTest(base=base.config):
                receipt = PolicyUses(base).finish("authority", "source")
                self.assertEqual(receipt.classes, ())
                for candidate in (
                    Config(block_sizes=[16]),
                    Config(block_sizes=[16], load_eviction_policies=[]),
                    Config(block_sizes=[32]),
                    Config(block_sizes=[16], load_eviction_policies=["first"]),
                ):
                    self.assertFalse(receipt.excludes(candidate))
        self.assertFalse(self.make_receipt().excludes(Config(block_sizes=[16])))

    def test_policy_free_neighbor_hook_keeps_geometry_neighbors(self):
        base = Config(block_sizes=[16])
        receipt = PolicyUses(base).finish("authority", "source")
        search = object.__new__(PatternSearch)
        search.kernel = object.__new__(BoundKernel)
        search.kernel._normalized_config_copy = lambda value: value
        search.config_gen = SimpleNamespace(
            overridden_flat_indices=set(),
            block_size_indices=[0],
            flat_spec=[SimpleNamespace(pattern_neighbors=lambda value: [32, 64])],
            unflatten=lambda flat: Config(block_sizes=flat),
        )
        search.num_neighbors_cap = 2
        search.eviction_policy_aliases_skipped = 0
        with patch(
            "helion._compiler.cute.eviction_relevance.receipt_for",
            return_value=receipt,
        ):
            self.assertEqual(search._generate_neighbors([16]), [[32], [64]])
        self.assertEqual(search.eviction_policy_aliases_skipped, 0)

    def test_global_policy_readers_and_missing_authority_disable(self):
        for flag in ({"cute_loop_vectorize": True}, {"cute_grid_work_order": "custom"}):
            self.assertIsNone(PolicyUses(config(**flag)).finish("authority", "source"))
        self.assertIsNone(PolicyUses(config()).finish(None, "source"))

    def test_context_does_not_escape_failure_or_nested_invocation(self):
        owner = object()
        with collect(config()) as outer:
            with self.assertRaises(RuntimeError), collect(config()) as inner:
                record_load(owner, 0, scalar_bytes=2)
                raise RuntimeError("failed source generation")
            record_load(owner, 1, scalar_bytes=4)
        self.assertEqual(len(inner.uses[0]), 1)
        self.assertEqual(outer.uses[0], [])
        self.assertEqual(len(outer.uses[1]), 1)
        record_load(owner, 0, scalar_bytes=2)
        self.assertEqual(outer.uses[0], [])

    def test_authority_allows_only_content_version_changes(self):
        tensor = torch.zeros(4)
        spec = SimpleNamespace(reusable_tensor_properties={"data_ptr", "storage_span"})
        bound = SimpleNamespace(
            settings=SimpleNamespace(backend="cute", static_shapes=True),
            env=SimpleNamespace(
                cute_fission_plan=None,
                cute_resolved_wrapper_plans={},
                process_group_name=None,
                runtime_input_specializations={"a": spec},
            ),
            host_function=SimpleNamespace(device_ir=SimpleNamespace(root_ids=[0])),
            _runtime_tensor_refs_by_name={"x": lambda: tensor},
        )
        facts = {
            "owners": [[1, 2, 3, [4], [1], 0, "fp32", "cpu", 0]],
            "graph": "original",
        }
        with patch(
            "helion.runtime.cute.source_batch._snapshot",
            side_effect=lambda *_: json.dumps(facts),
        ):
            original = authority(bound)
            self.assertIsNotNone(original)
            facts["owners"][0][-1] = 99
            self.assertEqual(authority(bound), original)
            for position in (0, 1, 2, 3, 4, 5, 6, 7):
                saved = copy.deepcopy(facts["owners"][0][position])
                facts["owners"][0][position] = "changed"
                self.assertNotEqual(authority(bound), original)
                facts["owners"][0][position] = saved
            facts["graph"] = "changed"
            self.assertNotEqual(authority(bound), original)
            spec.reusable_tensor_properties = set()
            self.assertIsNone(authority(bound))
        spec.reusable_tensor_properties = {"data_ptr"}
        with torch.inference_mode():
            inference_tensor = torch.zeros(4)
        bound._runtime_tensor_refs_by_name = {"x": lambda: inference_tensor}
        with patch(
            "helion.runtime.cute.source_batch._snapshot",
            side_effect=lambda _, tensors: tensors[0]._version,
        ):
            self.assertIsNone(authority(bound))

    def test_original_neighbor_hook_keeps_order_and_caps_after_alias_filter(self):
        class Fragment:
            def __init__(self, alternatives):
                self.alternatives = alternatives

            def pattern_neighbors(self, value):
                return self.alternatives

        class Generation:
            def __init__(self):
                self.overridden_flat_indices = set()
                self.block_size_indices = [1]
                self.flat_spec = [Fragment(["first", "l2_last"]), Fragment([32])]

            def unflatten(self, flat):
                result = config(["", flat[0]])
                result.config["block_sizes"] = [flat[1]]
                return result

        search = object.__new__(PatternSearch)
        search.kernel = object.__new__(BoundKernel)
        search.kernel._normalized_config_copy = lambda value: value
        search.config_gen = Generation()
        search.num_neighbors_cap = 2
        search.eviction_policy_aliases_skipped = 0
        with patch(
            "helion._compiler.cute.eviction_relevance.receipt_for",
            return_value=self.make_receipt(),
        ):
            self.assertEqual(
                search._generate_neighbors(["", 16]), [["first", 16], ["", 32]]
            )
        self.assertEqual(search.eviction_policy_aliases_skipped, 1)


if __name__ == "__main__":
    unittest.main()
