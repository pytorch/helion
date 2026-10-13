from __future__ import annotations

from dataclasses import replace

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_root_vector_cache import _args
from .test_cute_root_vector_cache import _config
from .test_cute_root_vector_cache import _sequence
from helion import exc
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_pointwise_residency import validate_pointwise_cache
from helion._compiler.generate_ast import GenerateAST


def test_pre_codegen_accepts_owned_root_vector_within_budget() -> None:
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
                assert plan.pointwise_cache is None
                cache = plan.root_vector_cache
                validate_pointwise_cache(plan, cache.allocated_bytes)
                for budget in (False, True, -1, 1.0, cache.allocated_bytes - 1):
                    with pytest.raises(exc.BackendUnsupported):
                        validate_pointwise_cache(plan, budget)
                with pytest.raises(exc.BackendUnsupported):
                    validate_pointwise_cache(
                        replace(plan, root_vector_cache=None), cache.allocated_bytes
                    )
                with pytest.raises(_UnsupportedChain, match="changed"):
                    validate_pointwise_cache(
                        replace(plan, root_vector_cache=replace(cache, phase=-1)),
                        cache.allocated_bytes,
                    )
            assert not context.host_statements and not context.module_statements
            assert not context.device_function.body
            assert not context.device_function.preamble
