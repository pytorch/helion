from __future__ import annotations

import ast
import inspect
import textwrap

from benchmarks.cute.recurrent_decode_inputs import build
from benchmarks.cute.recurrent_decode_inputs import cases
from examples.linear.recurrent_decode import kernel_args
import pytest
import torch

import helion
from helion.runtime.ref_mode import RefMode


def small_case(family: str, tokens: int = 1) -> dict[str, object]:
    case = dict(
        next(
            c for c in cases() if c["family"] == family and c.get("tokens", 1) == tokens
        )
    )
    case.update(
        device="cpu",
        batch=2,
        heads=2,
        value_heads=4 if family.startswith("gdn") else 2,
        key_dim=8,
        value_dim=8,
        state_dim=8,
        groups=1,
    )
    return case


def eager_kernel(kernel: helion.Kernel) -> helion.Kernel:
    """Spell native Cartesian indexing/store casts for PyTorch's eager evaluator.

    The production frontend is unchanged. Tile.index inside a mixed advanced
    index is otherwise interpreted as paired indexing by PyTorch, unlike the
    compiler's Cartesian tile semantics.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(kernel.fn)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    function.decorator_list = []

    class EagerIndex(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id in {
                "state",
                "initial_state",
                "final_state",
                "checkpoints",
                "a",
            }:

                class Tile(ast.NodeTransformer):
                    def visit_Attribute(self, value: ast.Attribute) -> ast.AST:
                        if (
                            value.attr == "index"
                            and isinstance(value.value, ast.Name)
                            and value.value.id == "tv"
                        ):
                            return value.value
                        return self.generic_visit(value)

                node.slice = Tile().visit(node.slice)
            return node

        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                target = node.targets[0]
                if isinstance(target.value, ast.Name) and target.value.id in {
                    "output",
                    "state",
                    "final_state",
                    "checkpoints",
                }:
                    node.value = ast.Call(
                        ast.Attribute(node.value, "to", ast.Load()),
                        [
                            ast.Attribute(
                                ast.Name(target.value.id, ast.Load()),
                                "dtype",
                                ast.Load(),
                            )
                        ],
                        [],
                    )
            return node

    tree = EagerIndex().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(kernel.fn.__globals__)
    exec(compile(tree, inspect.getsourcefile(kernel.fn), "exec"), namespace)
    return helion.kernel(
        namespace[function.name], ref_mode=RefMode.EAGER, fast_math=False
    )


@pytest.mark.parametrize(
    "family,tokens",
    [
        ("gdn-decode", 1),
        ("kda-decode", 1),
        ("gdn-mtp", 2),
        ("gdn-mtp", 4),
        ("mamba2-decode", 1),
        ("retention-decode", 1),
    ],
)
def test_small_raw_oracle_and_readonly(family: str, tokens: int) -> None:
    bundle = build(small_case(family, tokens), "helion")
    binding = bundle["objective_binding"]
    raw = binding["kernel_args"]
    prepared = kernel_args(family, raw)
    kernel = eager_kernel(next(iter(bundle["kernels"].values())))
    initial = {
        name: tensor.clone()
        for name, tensor in bundle["inputs"].items()
        if name in {"q", "k", "v", "mixed_qkv", "a", "b", "a_log", "dt_bias", "indices"}
    }
    retention_initial = raw[3].clone() if family == "retention-decode" else None
    before_state = binding["canonical_args"][7].clone() if family == "gdn-mtp" else None
    expected = bundle["reference"]()
    for _ in range(2):
        bundle["reset"]()
        kernel(*prepared)
        actual = tuple(binding["canonical_args"][i] for i in binding["result_indices"])
        if binding["state_pool"] is not None:
            actual += (binding["state_pool"][:1], binding["state_pool"][-1:])
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, atol=0.02, rtol=0.01)
        for name, before in initial.items():
            assert torch.equal(bundle["inputs"][name], before)
        if family == "gdn-mtp" and not raw[13]:
            assert torch.equal(raw[7], before_state)
        if family == "retention-decode":
            assert torch.equal(raw[3], retention_initial)


def test_programs_have_no_config_dispatch() -> None:
    for case in cases():
        kernel = next(
            iter(
                build(small_case(case["family"], case.get("tokens", 1)), "helion")[
                    "kernels"
                ].values()
            )
        )
        assert not kernel.configs
        assert not kernel.settings.autotune_seed_configs
        assert not kernel.settings.autotune_config_overrides


def test_projection_preserves_packed_alias_and_flags() -> None:
    for family in ("gdn-decode", "kda-decode"):
        bundle = build(small_case(family), "helion")
        raw = bundle["objective_binding"]["kernel_args"]
        prepared = kernel_args(family, raw)
        assert len(prepared) == 11
        assert (
            prepared[0].untyped_storage()._cdata == prepared[1].untyped_storage()._cdata
        )
        assert prepared[0].stride(0) == raw[0].stride(0)
        invalid = list(raw)
        invalid[12 if family == "gdn-decode" else 13] = 2
        with pytest.raises(AssertionError):
            kernel_args(family, tuple(invalid))


def test_preserves_original_cute_settings_migration() -> None:
    from helion.runtime.cute_structural_policy import CUTE_STRUCTURAL_SETTINGS

    for family in ("gdn-decode", "kda-decode", "mamba2-decode", "retention-decode"):
        bundle = build(small_case(family), "helion")
        actual = next(iter(bundle["kernels"].values()))
        # Exact canonical constructor followed by original discovery migration.
        original = helion.kernel(
            actual.fn,
            backend="cute",
            autotune_baseline_fn=bundle["objective_binding"]["kernel_reference"],
            autotune_baseline_atol=0.02,
            autotune_baseline_rtol=0.01,
        )
        original.settings.fast_math = False
        original.settings.autotune_baseline_fn = None
        original.settings.autotune_benchmark_subprocess = False
        assert actual.settings.static_shapes is original.settings.static_shapes is True
        assert actual.cute_structural_policy == original.cute_structural_policy
        for name in CUTE_STRUCTURAL_SETTINGS:
            assert (
                getattr(actual.settings, name)
                is getattr(original.settings, name)
                is True
            )
        assert actual.settings.autotune_baseline_fn is None
        assert actual.settings.autotune_baseline_atol == 0.02
        assert actual.settings.autotune_baseline_rtol == 0.01
