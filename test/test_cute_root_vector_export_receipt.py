from __future__ import annotations

from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_root_vector_cache import _args
from .test_cute_root_vector_cache import _config
from .test_cute_root_vector_cache import _sequence
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_root_exports import codegen_root_exports
from helion._compiler.cute.chained_root_vector_cache import VectorExportScope
from helion._compiler.cute.chained_root_vector_cache import publish_root_vector
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.generate_ast import GenerateAST


def test_actual_vector_publication_export_keeps_owned_scope() -> None:
    with _cpu_codegen():
        bound = _sequence._bind_isolated(_args())
        config = bound.config_spec.normalized_config(_config())
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            context = GenerateAST(bound.host_function, config)
            with context.device_function:
                bound.env.backend.pre_codegen(
                    context.codegen_graphs,
                    config,
                    context.device_function.tile_strategy,
                )
                plan = context.device_function.cute_state.chained_matmul_plan
                assert plan is not None and plan.root_vector_cache is not None
                cache = plan.root_vector_cache
                context.current_root_graph_info = cache.root
                boundaries = {}
                lines = []
                publication = publish_root_vector(
                    context,
                    plan,
                    lines,
                    boundaries,
                    ScratchLayouts.for_plan("row_major", plan),
                )
                assert publication is not None
                expression = _Expression(context, plan, boundaries)
                with pytest.raises(_UnsupportedChain, match="precedes its owned scope"):
                    expression.value(cache.node, ("foreign_index",))
                # Exercise the real export helper and its actual read receipt;
                # no full kernel visitor, to_code, or SDK compilation is needed.
                exported = codegen_root_exports(context, plan, boundaries, [], [])
                lines.extend(exported)
                publication.finish(boundaries)
                assert len(publication.reads) == 1
                read = publication.reads[0]
                assert isinstance(read.origin, VectorExportScope)
                assert read.origin.export.store.args[2] is cache.node
                assert read.origin.extent == cache.extent
                assert f"{cache.name}[{read.origin.prefix}]" in "\n".join(exported)
                assert any(
                    f"if {read.origin.prefix} < {cache.extent}:" in line
                    for line in exported
                )
                prefix = tuple(lines[: len(publication.prefix)])
                with (
                    patch.dict(boundaries, {cache.node: "foreign_vector"}),
                    pytest.raises(_UnsupportedChain, match="lost or unused"),
                ):
                    publication.finish(boundaries)
                lines[len(publication.prefix) - 1] = "foreign_join()"
                with pytest.raises(_UnsupportedChain, match="prefix or team changed"):
                    publication.finish(boundaries)
                lines[: len(publication.prefix)] = prefix
                publication.finish(boundaries)
            assert not context.host_statements and not context.module_statements
            assert not context.device_function.body
            assert not context.device_function.preamble
