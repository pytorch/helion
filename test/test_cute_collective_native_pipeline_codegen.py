"""Real CPU-bound compiler-AST regressions for the native operand pipeline."""

from __future__ import annotations

import ast
import copy
import unittest
from unittest.mock import patch

import torch

import helion
from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import LoopType
from helion._compiler.ast_extension import get_wrapper_cls
from helion._compiler.ast_read_writes import ast_rename
from helion._compiler.cute.collective_tcgen05 import CollectiveTcgen05Plan
from helion._compiler.cute.scalar_recipe import _clone
from helion._compiler.source_location import UnknownLocation
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    cute_structural_policy="auto",
)
def _computed_pipeline(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            av = a[row, reduction].to(torch.float32).to(a.dtype)
            bv = b[reduction, column].to(torch.float32).to(b.dtype)
            acc = torch.addmm(acc, av, bv)
        value = torch.relu(acc + bias[column].to(torch.float32))
        value = value + residual[row, column].to(torch.float32) * 0.5
        out[row, column] = value.to(out.dtype)


def _config(stages: int) -> helion.Config:
    # The real saved-stage2 geometry/recipe family, with four input load sites.
    return helion.Config.from_dict(
        {
            "block_sizes": [128, 32, 128],
            "num_threads": [4, 32, 1],
            "loop_orders": [[0, 1]],
            "cute_vector_widths": [1, 1, 1],
            "cute_lane_layouts": ["blocked", "strided", "blocked"],
            "cute_collective_mma": True,
            "cute_collective_compute": "tcgen05",
            "cute_collective_copy": "async_cached",
            "cute_collective_recipe": "vector_unrolled",
            "cute_collective_epilogue": "vector",
            "cute_collective_stages": stages,
            "cute_collective_tmem_a": True,
            "cute_collective_tmem_seed": False,
            "cute_collective_native_seeded": False,
            "cute_proven_bounds": True,
            "cute_register_chain": False,
            "cute_pdl": False,
            "cute_min_blocks_per_mp": 0,
            "load_eviction_policies": ["", "", "", ""],
        }
    )


def _kernel(source: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


def _epilogue(source: str) -> list[str]:
    return [
        ast.unparse(node)
        for node in ast.walk(_kernel(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("epilogue_vector_slot")
    ]


class NativePipelineCompilerAST(unittest.TestCase):
    def test_real_extended_clone_preserves_location_and_metadata(self):
        location = UnknownLocation()
        marker = object()
        wrapper = get_wrapper_cls(ast.Name)
        node = wrapper(
            id="k",
            ctx=ast.Load(),
            _location=location,
            _type_info=marker,
            _loop_type=LoopType.DEVICE,
            _is_kernel_call=True,
            _root_id=19,
            _compiler_inserted=True,
        )
        with self.assertRaisesRegex(TypeError, "_location"):
            copy.deepcopy(node)
        cloned = ast_rename(_clone(node), {"k": "prefetch_k"})
        self.assertEqual(node.id, "k")
        self.assertEqual(cloned.id, "prefetch_k")
        self.assertIsNot(cloned, node)
        self.assertIsInstance(cloned, ExtendedAST)
        self.assertIs(cloned._location, location)
        self.assertIs(cloned._type_info, marker)
        self.assertEqual(cloned._loop_type, LoopType.DEVICE)
        self.assertTrue(cloned._is_kernel_call)
        self.assertEqual(cloned._root_id, 19)
        self.assertTrue(cloned._compiler_inserted)

    def test_real_cpu_bound_saved_style_stage2_to_code(self):
        for dtype in (torch.float16, torch.bfloat16):
            with (
                self.subTest(dtype=dtype),
                mock_cuda_unavailable(),
                forbid_cuda_work(),
            ):
                args = (
                    torch.empty((256, 768), dtype=dtype),
                    torch.empty_strided((768, 384), (1, 768), dtype=dtype),
                    torch.empty((384,), dtype=dtype),
                    torch.empty((256, 384), dtype=dtype),
                    torch.empty((256, 384), dtype=dtype),
                )
                # CPU source emission models capacity; it does not query a GPU.
                with patch(
                    "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
                    return_value=232448,
                ):
                    bound = cpu_bind(_computed_pipeline, args)
                    control = bound.to_code(_config(1))
                    self.assertIn("OperandSource.TMEM", control)
                    self.assertNotIn("_next_k", control)
                    with patch(
                        "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
                        return_value=36863,
                    ):
                        limited = bound.to_code(_config(2))
                    self.assertNotIn("_next_k", limited)
                    # Negative control proves the real compiler AST reaches the
                    # exact v1 failure, rather than testing an unused emitter.
                    with (
                        patch(
                            "helion._compiler.cute.collective_tcgen05._clone",
                            side_effect=copy.deepcopy,
                        ),
                        self.assertRaisesRegex(TypeError, "_location"),
                    ):
                        bound.to_code(_config(2))
                    observed = []
                    original = CollectiveTcgen05Plan.pipeline_loop

                    def record(
                        plan,
                        *arguments,
                        _original=original,
                        _observed=observed,
                        **options,
                    ):
                        nodes = [
                            node
                            for value in arguments
                            for node in (
                                ast.walk(value) if isinstance(value, ast.AST) else []
                            )
                            if isinstance(node, ExtendedAST)
                        ]
                        self.assertTrue(
                            nodes, "real ExtendedAST did not reach the pipeline"
                        )
                        before = [
                            (node, node._location, node._root_id) for node in nodes
                        ]
                        result = _original(plan, *arguments, **options)
                        for node, owner, root in before:
                            self.assertIs(node._location, owner)
                            self.assertEqual(node._root_id, root)
                        _observed.extend(nodes)
                        return result

                    with patch.object(CollectiveTcgen05Plan, "pipeline_loop", record):
                        source = bound.to_code(_config(2))
                    self.assertTrue(observed)
                    self.assertIn("_next_k", source)
                    self.assertIn("_b_stage", source)
                    self.assertIn("_b_prefill", source)
                    self.assertIn("_b_current", source)
                    self.assertIn("_b_prefetch", source)
                    self.assertIn("_rb_current", source)
                    self.assertIn("OperandSource.TMEM", source)
                    self.assertEqual(_epilogue(control), _epilogue(source))
                    loop = next(
                        node
                        for node in ast.walk(_kernel(source))
                        if isinstance(node, ast.For)
                        and any(
                            isinstance(child, ast.Name) and child.id.endswith("_next_k")
                            for child in ast.walk(node)
                        )
                    )
                    parts = [ast.unparse(node) for node in loop.body]
                    issue = next(
                        i for i, text in enumerate(parts) if "cute.gemm" in text
                    )
                    prefetch = next(
                        i
                        for i, text in enumerate(parts)
                        if isinstance(loop.body[i], ast.If)
                        and "_next_k" in text
                        and "_a_values" in text
                    )
                    wait = next(
                        i for i, text in enumerate(parts) if "mbarrier_wait" in text
                    )
                    publish = next(
                        i
                        for i, text in enumerate(parts)
                        if isinstance(loop.body[i], ast.If)
                        and "_a_store_copy" in text
                        and i > wait
                    )
                    self.assertLess(issue, prefetch)
                    self.assertLess(prefetch, wait)
                    self.assertLess(wait, publish)


if __name__ == "__main__":
    unittest.main()
