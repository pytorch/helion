from __future__ import annotations

from collections import Counter
import functools
import math
import unittest

from benchmarks.cute import attngym_figure as figure
from benchmarks.cute import attngym_variants as agv
from benchmarks.cute.attngym_report import FIGURE_GROUPS
import torch
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel

CPU = torch.device("cpu")


def _tiny_variant(name="noop", *, b=2, hq=4, hkv=2, m=4, n=4, d=2, dv=3, params=None):
    extras = {"modifier_seed": 7, **(params or {})}
    return agv.Variant(
        name=name,
        b=b,
        hq=hq,
        hkv=hkv,
        m=m,
        n=n,
        d=d,
        dv=dv,
        dtype=torch.float32,
        make_mods=functools.partial(figure._mods, name, b, hq, m, n, extras),
        extras=extras,
        suite="figure",
        shape_id=f"{name}.tiny",
        mask_batch=b if name == "vsa" else 1,
        mask_heads=hq if name == "vsa" else 1,
    )


def _mask(name, m, n, params):
    score_mod, mask_mod = figure._mods(name, 1, 1, m, n, params, CPU)
    assert score_mod is None
    assert mask_mod is not None
    return mask_mod


def _visible_keys(mask_mod, query, n):
    zero = torch.tensor(0)
    mask = mask_mod(zero, zero, torch.tensor(query), torch.arange(n))
    return set(mask.nonzero().flatten().tolist())


class TestAttentionGymFigure(unittest.TestCase):
    def test_manifest_covers_29_variants_with_three_unique_cases_each(self):
        cases = list(figure.CASES.values())
        names = [name for group, group_names in FIGURE_GROUPS for name in group_names]
        self.assertEqual(len(cases), 87)
        self.assertEqual(len({case.shape_id for case in cases}), 87)
        self.assertEqual(Counter(case.name for case in cases), dict.fromkeys(names, 3))
        self.assertEqual(list(dict.fromkeys(case.name for case in cases)), names)
        for case in cases:
            with self.subTest(shape_id=case.shape_id):
                self.assertIs(figure.CASES[case.shape_id], case)
                self.assertEqual((case.b, case.hq, case.dtype), (4, 16, torch.bfloat16))
                self.assertEqual(case.suite, "figure")
                self.assertEqual(case.hq % case.hkv, 0)
                self.assertEqual(case.b % case.mask_batch, 0)
                self.assertEqual(case.hq % case.mask_heads, 0)
                self.assertGreater(min(case.m, case.n, case.d, case.value_dim), 0)
                if case.name in {"natten2d", "sta2d", "sta3d"}:
                    self.assertEqual(case.m, math.prod(case.extras["canvas"]))
                    self.assertEqual(case.m, case.n)
                if case.name in {"sta2d", "sta3d"}:
                    for canvas, kernel, tile in zip(
                        case.extras["canvas"],
                        case.extras["kernel"],
                        case.extras["tile"],
                        strict=True,
                    ):
                        self.assertEqual(canvas % tile, 0)
                        self.assertEqual(kernel % tile, 0)
                        self.assertLessEqual(kernel, canvas)
                if case.name.startswith("svg-"):
                    self.assertEqual(
                        case.m,
                        case.extras["prompt"]
                        + case.extras["frames"] * case.extras["tokens_per_frame"],
                    )
                if case.name.startswith("jetspec-"):
                    self.assertEqual(case.n, case.m + case.extras["prefix"])
                if case.name in {"document", "shared-prefix"}:
                    self.assertEqual(sum(case.extras["document_lengths"]), case.m)
                    self.assertGreater(min(case.extras["document_lengths"]), 0)
                if case.name == "flamingo":
                    self.assertEqual(case.m, 2 * case.n)
                    self.assertEqual(case.n % case.extras["images"], 0)
                if case.name == "vsa":
                    self.assertEqual(case.m % case.extras["tile_numel"], 0)
                    self.assertEqual((case.mask_batch, case.mask_heads), (4, 16))
                if case.name == "mla-causal":
                    self.assertEqual((case.hkv, case.d, case.value_dim), (1, 192, 128))
                    self.assertEqual(
                        case.d, case.extras["latent_dim"] + case.extras["rope_dim"]
                    )
        manifest = figure.manifest()
        self.assertEqual(
            {item["shape_id"] for item in manifest["cases"]}, set(figure.CASES)
        )
        self.assertIn("not the original B200", manifest["description"])

    def test_visible_pair_count_broadcasts_all_query_key_batch_head_axes(self):
        def mask(b, h, q, kv):
            # One query per batch/head sees all keys; the returned tensor has
            # a singleton key axis because the mask does not depend on kv.
            return q == (b + h) % 3

        def make_mods(device):
            return None, mask

        variant = agv.Variant(
            "tiny-dependent-mask",
            b=2,
            hq=4,
            hkv=2,
            m=3,
            n=5,
            d=2,
            dv=3,
            dtype=torch.float32,
            make_mods=make_mods,
            suite="figure",
            mask_batch=2,
            mask_heads=4,
        )
        # 2 batches * 4 query heads * 1 visible row * 5 keys = 40 pairs.
        self.assertEqual(agv._pair_count(variant, CPU), 40)
        # QK uses D=2; PV uses Dv=3, independent of GQA's KV-head count.
        self.assertEqual(agv.variant_flops(variant, CPU), 400.0)

    def test_pair_count_counts_broadcast_batch_and_head_copies(self):
        def mask(b, h, q, kv):
            return q >= 0

        def make_mods(device):
            return None, mask

        variant = _tiny_variant(b=3, hq=4, hkv=2, m=3, n=5)
        variant.make_mods = make_mods
        self.assertEqual(agv._pair_count(variant, CPU), 3 * 4 * 3 * 5)

    def test_seeded_inputs_have_asymmetric_value_dimensions_and_preserve_rng(self):
        variant = _tiny_variant(m=3, n=5, d=2, dv=3)
        state = torch.random.get_rng_state().clone()
        first = agv.make_qkv(variant, CPU, seed=17)
        repeated = agv.make_qkv(variant, CPU, seed=17)
        changed = agv.make_qkv(variant, CPU, seed=18)
        self.assertEqual(
            [tuple(tensor.shape) for tensor in first],
            [(2, 4, 3, 2), (2, 2, 5, 2), (2, 2, 5, 3)],
        )
        for tensor, same, different in zip(first, repeated, changed, strict=True):
            self.assertTrue(torch.equal(tensor, same))
            self.assertFalse(torch.equal(tensor, different))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(_tiny_variant(d=7, dv=None).value_dim, 7)

    def test_reference_returns_zero_for_empty_rows_and_uses_correct_gqa_values(self):
        def mask(b, h, q, kv):
            return (q > 0) & (kv < q)

        variant = _tiny_variant(b=1, hq=2, hkv=1, m=3, n=2, d=2, dv=3)
        q = torch.zeros(1, 2, 3, 2)
        k = torch.zeros(1, 1, 2, 2)
        v = torch.tensor([[[[3.0, 6.0, 9.0], [9.0, 12.0, 15.0]]]])
        expected = torch.tensor(
            [[[[0.0, 0.0, 0.0], [3.0, 6.0, 9.0], [6.0, 9.0, 12.0]]]]
        ).expand(1, 2, 3, 3)
        for chunk in (1, 3):
            with self.subTest(chunk=chunk):
                actual = agv.reference_output(
                    variant, q, k, v, None, mask, q_chunk=chunk
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertTrue(torch.isfinite(actual).all())

    def test_jetspec_tree_has_only_prefix_and_root_to_self_paths(self):
        mask = _mask(
            "jetspec-tree", 6, 8, {"prefix": 2, "parents": [-1, 0, 0, 1, 1, 2]}
        )
        expected = (
            {0, 1, 2},
            {0, 1, 2, 3},
            {0, 1, 2, 4},
            {0, 1, 2, 3, 5},
            {0, 1, 2, 3, 6},
            {0, 1, 2, 4, 7},
        )
        for query, keys in enumerate(expected):
            with self.subTest(query=query):
                self.assertEqual(_visible_keys(mask, query, 8), keys)

    def test_jetspec_training_never_crosses_draft_blocks(self):
        mask = _mask("jetspec-training", 6, 8, {"prefix": 2, "block_size": 3})
        expected = (
            {0, 1, 2},
            {0, 1, 2, 3},
            {0, 1, 2, 3, 4},
            {0, 1, 5},
            {0, 1, 5, 6},
            {0, 1, 5, 6, 7},
        )
        for query, keys in enumerate(expected):
            with self.subTest(query=query):
                self.assertEqual(_visible_keys(mask, query, 8), keys)

    def test_block_diffusion_visibility_across_noised_and_clean_blocks(self):
        # Three two-token blocks in each half: noised [0, 6), clean [6, 12).
        # Explicit key sets cover both halves and earlier/same/later blocks.
        mask = _mask("block-diffusion", 12, 12, {"block_size": 2})
        expected = (
            {0, 1},
            {0, 1},
            {2, 3, 6, 7},
            {2, 3, 6, 7},
            {4, 5, 6, 7, 8, 9},
            {4, 5, 6, 7, 8, 9},
            {6, 7},
            {6, 7},
            {6, 7, 8, 9},
            {6, 7, 8, 9},
            {6, 7, 8, 9, 10, 11},
            {6, 7, 8, 9, 10, 11},
        )
        for query, keys in enumerate(expected):
            with self.subTest(query=query):
                self.assertEqual(_visible_keys(mask, query, 12), keys)

    def test_sta2d_uses_tile_major_order_and_even_window_borders(self):
        mask = _mask(
            "sta2d", 24, 24, {"canvas": [4, 6], "kernel": [4, 4], "tile": [2, 2]}
        )
        left = {tile * 4 + token for tile in (0, 1, 3, 4) for token in range(4)}
        right = {tile * 4 + token for tile in (1, 2, 4, 5) for token in range(4)}
        self.assertEqual(_visible_keys(mask, 0, 24), left)
        self.assertEqual(_visible_keys(mask, 3, 24), left)
        self.assertEqual(_visible_keys(mask, 8, 24), right)
        self.assertEqual(_visible_keys(mask, 23, 24), right)

    def test_sta3d_uses_time_height_width_tile_coordinates(self):
        mask = _mask(
            "sta3d",
            96,
            96,
            {"canvas": [4, 4, 6], "kernel": [4, 4, 4], "tile": [2, 2, 2]},
        )
        left = {
            tile * 8 + token for tile in (0, 1, 3, 4, 6, 7, 9, 10) for token in range(8)
        }
        right = {
            tile * 8 + token
            for tile in (1, 2, 4, 5, 7, 8, 10, 11)
            for token in range(8)
        }
        self.assertEqual(_visible_keys(mask, 0, 96), left)
        self.assertEqual(_visible_keys(mask, 7, 96), left)
        self.assertEqual(_visible_keys(mask, 95, 96), right)

    def test_natten_neighborhood_is_recentered_at_image_edges(self):
        mask = _mask("natten2d", 25, 25, {"canvas": [5, 5], "kernel": [3, 3]})
        self.assertEqual(_visible_keys(mask, 0, 25), {0, 1, 2, 5, 6, 7, 10, 11, 12})
        self.assertEqual(
            _visible_keys(mask, 24, 25), {12, 13, 14, 17, 18, 19, 22, 23, 24}
        )

    def test_svg_spatial_rounding_sink_and_temporal_frame_positions(self):
        params = {
            "prompt": 2,
            "frames": 4,
            "tokens_per_frame": 3,
            "width": 2,
            "round_width": 4,
            "attn_sink": True,
        }
        spatial = _mask("svg-spatial", 14, 14, params)
        temporal = _mask("svg-temporal", 14, 14, params)
        self.assertEqual(_visible_keys(spatial, 1, 14), set(range(14)))
        # Rounded radius is 4; video frame zero remains an attention sink.
        self.assertEqual(_visible_keys(spatial, 8, 14), set(range(12)))
        self.assertEqual(
            _visible_keys(spatial, 13, 14), {0, 1, 2, 3, 4, 10, 11, 12, 13}
        )
        self.assertEqual(_visible_keys(temporal, 1, 14), set(range(14)))
        self.assertEqual(_visible_keys(temporal, 8, 14), {0, 1, 2, 5, 8, 11})
        self.assertEqual(_visible_keys(temporal, 9, 14), {0, 1, 3, 6, 9, 12})

    def test_vsa_has_fixed_distinct_kv_tiles_and_seeded_batch_head_layouts(self):
        variant = _tiny_variant("vsa", m=8, n=8, params={"tile_numel": 2, "top_k": 2})
        _, first = variant.make_mods(CPU)
        _, repeated = variant.make_mods(CPU)
        b = torch.arange(2)[:, None, None, None]
        h = torch.arange(4)[None, :, None, None]
        q = torch.arange(8)[None, None, :, None]
        k = torch.arange(8)[None, None, None, :]
        mask = first(b, h, q, k)
        self.assertEqual(tuple(mask.shape), (2, 4, 8, 8))
        self.assertTrue(torch.equal(mask, repeated(b, h, q, k)))
        self.assertTrue((mask.sum(dim=-1) == 4).all())
        self.assertTrue(torch.equal(mask[:, :, ::2], mask[:, :, 1::2]))
        self.assertTrue(torch.equal(mask[:, :, :, ::2], mask[:, :, :, 1::2]))
        self.assertEqual(agv._pair_count(variant, CPU), 2 * 4 * 8 * 4)

    def test_graphormer_ring_paths_choose_clockwise_ties_and_label_wrap_edges(self):
        params = {"max_distance": 1, "path_len": 3, "edge_types": 8, "modifier_seed": 7}
        distances, spatial_bias, lengths, paths, edge_bias = figure._graph_tables(
            2, 3, 6, params, CPU
        )
        self.assertEqual(distances[0, 0].tolist(), [0, 1, 2, 2, 2, 1])
        self.assertEqual(lengths[0, 0].tolist(), [0, 1, 2, 3, 2, 1])
        self.assertEqual(paths[0, 0, 3].tolist(), [0, 1, 2])
        self.assertEqual(paths[0, 3, 0].tolist(), [3, 4, 5])
        self.assertEqual(paths[0, 0, 4, :2].tolist(), [5, 4])
        self.assertEqual(paths[0, 0, 5, 0].item(), 5)
        self.assertEqual(paths[0, 5, 0, 0].item(), 5)
        self.assertEqual(paths[1, 0, 3].tolist(), [3, 4, 5])
        repeated = figure._graph_tables(2, 3, 6, params, CPU)
        self.assertTrue(torch.equal(spatial_bias, repeated[1]))
        self.assertTrue(torch.equal(edge_bias, repeated[4]))

    def test_graphormer_score_bias_averages_only_valid_path_edges(self):
        params = {"max_distance": 1, "path_len": 3, "edge_types": 8, "modifier_seed": 7}
        distances, spatial_bias, lengths, paths, edge_bias = figure._graph_tables(
            2, 3, 6, params, CPU
        )
        spatial, _ = figure._mods("graphormer-spatial", 2, 3, 6, 6, params, CPU)
        edge, _ = figure._mods("graphormer-edge", 2, 3, 6, 6, params, CPU)
        score, b, h, q = (
            torch.tensor(1.25),
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(0),
        )
        self.assertEqual(
            spatial(score, b, h, q, torch.tensor(4)), score + spatial_bias[2, 2]
        )
        # Batch one labels 0->5->4 with types 0 and 7, respectively.
        expected = score + (edge_bias[2, 0] + edge_bias[2, 8 + 7]) / 2
        torch.testing.assert_close(edge(score, b, h, q, torch.tensor(4)), expected)
        torch.testing.assert_close(edge(score, b, h, q, q), score, atol=0, rtol=0)

    def test_additive_dense_bias_matches_attention_math_with_gqa_and_asymmetric_dv(
        self,
    ):
        specifications = (
            ("causal", {}),
            ("alibi", {}),
            ("rel-bias", {}),
            ("vsa", {"tile_numel": 2, "top_k": 1}),
            ("flamingo", {"images": 2, "visible_images": 1}),
            ("graphormer-spatial", {"max_distance": 1, "path_len": 2, "edge_types": 3}),
            ("graphormer-edge", {"max_distance": 1, "path_len": 2, "edge_types": 3}),
        )
        for name, params in specifications:
            with self.subTest(name=name):
                variant = _tiny_variant(
                    name, n=2 if name == "flamingo" else 4, params=params
                )
                q, k, v = agv.make_qkv(variant, CPU, seed=17)
                score_mod, mask_mod = variant.make_mods(CPU)
                reference = agv.reference_output(
                    variant, q, k, v, score_mod, mask_mod, q_chunk=2
                )
                bias = figure.dense_bias(variant, CPU)
                with sdpa_kernel(SDPBackend.MATH):
                    actual = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, enable_gqa=True
                    )
                torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)

    def test_alibi_dense_bias_keeps_per_head_slopes(self):
        variant = _tiny_variant("alibi", hq=2, hkv=1, m=2, n=3)
        expected = torch.tensor(
            [
                [
                    [[0, 1 / 16, 2 / 16], [-1 / 16, 0, 1 / 16]],
                    [[0, 1 / 256, 2 / 256], [-1 / 256, 0, 1 / 256]],
                ]
            ]
        )
        torch.testing.assert_close(
            figure.dense_bias(variant, CPU), expected, atol=0, rtol=0
        )

    def test_nonlinear_modifiers_cannot_use_an_additive_sdpa_bias(self):
        for name in ("softcap", "sigmoid-act", "gemma2"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "not expressible"),
            ):
                figure.dense_bias(_tiny_variant(name), CPU)


if __name__ == "__main__":
    unittest.main()
