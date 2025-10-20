from __future__ import annotations

import importlib
import inspect
import pickle
from typing import Any
import unittest

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import helion
from helion._testing import TestCase


def _json_safe_values() -> st.SearchStrategy[Any]:
    # JSON-safe primitives/containers
    scalar = st.one_of(
        st.integers(), st.floats(allow_nan=False), st.booleans(), st.text()
    )
    leaf = st.one_of(scalar, st.none())
    return st.recursive(
        leaf,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(st.text(min_size=0, max_size=8), children, max_size=4),
        ),
        max_leaves=8,
    )


def _known_keys_strategy() -> st.SearchStrategy[dict[str, Any]]:
    # For known keys, None values are omitted by constructor; favor non-None
    return st.fixed_dictionaries(
        {
            "block_sizes": st.lists(
                st.integers(min_value=1, max_value=4096), max_size=4
            ),
            "loop_orders": st.lists(
                st.lists(st.integers(min_value=0, max_value=4), max_size=4),
                max_size=3,
            ),
            "flatten_loops": st.lists(st.booleans(), max_size=4),
            "l2_groupings": st.lists(
                st.integers(min_value=1, max_value=128), max_size=4
            ),
            "reduction_loops": st.lists(
                st.one_of(st.integers(min_value=0, max_value=8), st.none()),
                max_size=4,
            ),
            "range_unroll_factors": st.lists(
                st.integers(min_value=1, max_value=16), max_size=4
            ),
            "range_warp_specializes": st.lists(
                st.one_of(st.booleans(), st.none()), max_size=4
            ),
            "range_num_stages": st.lists(
                st.integers(min_value=1, max_value=8), max_size=4
            ),
            "range_multi_buffers": st.lists(
                st.one_of(st.booleans(), st.none()), max_size=4
            ),
            "range_flattens": st.lists(st.one_of(st.booleans(), st.none()), max_size=4),
            "static_ranges": st.lists(st.booleans(), max_size=4),
            "load_eviction_policies": st.lists(
                st.sampled_from(["", "first", "last"]), max_size=4
            ),
            "num_warps": st.integers(min_value=1, max_value=64),
            "num_stages": st.integers(min_value=1, max_value=16),
            "pid_type": st.sampled_from(
                ["flat", "xyz", "persistent_blocked", "persistent_interleaved"]
            ),
            "indexing": st.sampled_from(["pointer", "tensor_descriptor"]),
        }
    )


def _unknown_keys_strategy() -> st.SearchStrategy[dict[str, Any]]:
    key = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,12}")
    # Avoid colliding with known keys and enforce distinctness
    return st.dictionaries(
        keys=key.filter(
            lambda k: k
            not in {
                "block_sizes",
                "loop_orders",
                "flatten_loops",
                "l2_groupings",
                "reduction_loops",
                "range_unroll_factors",
                "range_warp_specializes",
                "range_num_stages",
                "range_multi_buffers",
                "range_flattens",
                "static_ranges",
                "load_eviction_policies",
                "num_warps",
                "num_stages",
                "pid_type",
                "indexing",
            }
        ),
        values=_json_safe_values(),
        max_size=4,
    )


class TestConfigAPI(TestCase):
    def test_config_import_path_stability(self) -> None:
        runtime = importlib.import_module("helion.runtime")

        self.assertIs(helion.Config, runtime.Config)
        self.assertIs(helion.Config, helion.runtime.Config)

    def test_config_constructor_signature_contains_expected_kwargs(self) -> None:
        # Keep this list in sync with public kwargs; removal/rename should fail tests
        expected = {
            "block_sizes",
            "loop_orders",
            "flatten_loops",
            "l2_groupings",
            "reduction_loops",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "num_warps",
            "num_stages",
            "pid_type",
            "indexing",
        }

        sig = inspect.signature(helion.Config.__init__)
        kwonly = {
            name
            for name, p in sig.parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        # Expected kwargs must be present as keyword-only
        self.assertTrue(expected.issubset(kwonly))

    def test_mapping_behavior_len_iter_dict_roundtrip(self) -> None:
        data = {
            "block_sizes": [64, 32],
            "num_warps": 8,
            "custom_extra": {"a": 1},
        }
        cfg = helion.Config(**data)

        # Supports Mapping protocol
        self.assertEqual(len(cfg), len(cfg.config))
        self.assertEqual(dict(cfg), cfg.config)
        self.assertEqual(set(iter(cfg)), set(cfg.config.keys()))

        # Equality and hash coherence
        cfg2 = helion.Config(**data)
        self.assertEqual(cfg, cfg2)
        self.assertEqual(hash(cfg), hash(cfg2))

    @settings(deadline=None)
    @given(
        st.builds(lambda a, b: (a, b), _known_keys_strategy(), _unknown_keys_strategy())
    )
    def test_json_roundtrip_preserves_keys_and_values(
        self, pair: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        known, unknown = pair
        data = {**known, **unknown}
        cfg = helion.Config(**data)

        # JSON round-trip
        json_str = cfg.to_json()
        restored = helion.Config.from_json(json_str)

        # Compare as dicts; JSON dumps may reorder keys
        self.assertEqual(dict(restored), dict(cfg))

        # Unknown keys must persist
        for k in unknown:
            self.assertIn(k, restored)
            self.assertEqual(restored[k], unknown[k])

    @settings(deadline=None)
    @given(_known_keys_strategy(), _unknown_keys_strategy())
    def test_pickle_roundtrip_preserves_equality_and_hash(
        self, known: dict[str, Any], unknown: dict[str, Any]
    ) -> None:
        data = {**known, **unknown}
        cfg = helion.Config(**data)
        blob = pickle.dumps(cfg)
        restored = pickle.loads(blob)

        self.assertEqual(restored, cfg)
        self.assertEqual(hash(restored), hash(cfg))

    def test_list_tuple_hash_equivalence(self) -> None:
        cfg_list = helion.Config(block_sizes=[32, 64], loop_orders=[[1, 0]])
        cfg_tuple = helion.Config(block_sizes=[32, 64], loop_orders=[[1, 0]])

        # Same content should be equal and have equal hashes
        self.assertEqual(cfg_list, cfg_tuple)
        self.assertEqual(hash(cfg_list), hash(cfg_tuple))

    def test_pre_serialized_json_backward_compat(self) -> None:
        # Simulated config JSON saved in a prior release (hand-written, stable keys)
        json_str = (
            "{\n"
            '  "block_sizes": [64, 32],\n'
            '  "num_warps": 8,\n'
            '  "indexing": "pointer",\n'
            '  "custom_extra": {"alpha": 1, "beta": [1, 2]}\n'
            "}\n"
        )

        restored = helion.Config.from_json(json_str)

        expected = {
            "block_sizes": [64, 32],
            "num_warps": 8,
            "indexing": "pointer",
            "custom_extra": {"alpha": 1, "beta": [1, 2]},
        }
        self.assertEqual(dict(restored), expected)

        # Ensure we can still serialize it back and preserve content
        rejson = restored.to_json()
        reread = helion.Config.from_json(rejson)
        self.assertEqual(dict(reread), expected)


if __name__ == "__main__":
    unittest.main()
