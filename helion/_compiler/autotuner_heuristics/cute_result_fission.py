"""Register the result-fission proof and transfer its seeds last.

The schedule is off in the public default and in random samples; it enters the
search only through seeds appended after every legacy seed and carrier
(``transferred_seed_configs``), so existing seed prefixes and promoted
defaults are unchanged.  Each seed keeps the base default config, turns the
schedule on, and takes its producer GEMM schedule from one of the kernel's own
native TCGEN05 compiler seeds -- admitted only when the producer's one-C-output
normalizer accepts it exactly -- with the coalesced scalar consumer default
and, where the output shape admits whole-warp 16-byte vectors, an N-vectorized
consumer twin.  Every admitted producer gets both consumer layouts (at most
``MAX_PRODUCER_SEEDS`` producers, so at most twice as many seeds).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...exc import InvalidConfig
from ...runtime.config import Config
from ..cute.result_fission_config import CONSUMER_BLOCK_SIZES_KEY
from ..cute.result_fission_config import CONSUMER_NUM_THREADS_KEY
from ..cute.result_fission_config import CONSUMER_VECTOR_WIDTH_KEY
from ..cute.result_fission_config import RESULT_FISSION_KEY
from ..cute.result_fission_config import default_consumer_block_sizes
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

MAX_PRODUCER_SEEDS = 8


log: logging.Logger = logging.getLogger(__name__)


class CuteResultFissionHeuristic(AutotunerHeuristic):
    name = "cute_result_fission"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"config_num_sm"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.result_fission import analyze_result_fission

        spec = env.config_spec
        spec.cute_result_fission_available = False
        spec.cute_result_fission_shape = None
        spec.cute_result_fission_producer_facts = None
        if device_ir.host_function is None:
            return frozenset()
        from ..cute.result_fission_codegen import producer_admission

        # An optional strategy never fails a bind the fused path supports:
        # any failure of its registration-time proof or of the producer's
        # admission leaves it off.
        try:
            with device_ir.host_function:
                # Registration reads the already registered live alias
                # classifiers (the shared fanout coverage contract); codegen
                # re-proves on the final bound facts and fails loudly if lost.
                proof = analyze_result_fission(env, device_ir, allow_unbound=True)
                if proof is None:
                    return frozenset()
                admission = producer_admission(proof, env)
        except Exception:
            log.debug("result-fission registration failed", exc_info=True)
            return frozenset()
        if admission.max_ab_stages and any(admission.max_ab_stages.values()):
            spec.cute_result_fission_available = True
            spec.cute_result_fission_shape = (proof.m, proof.n, proof.k)
            spec.cute_result_fission_producer_facts = admission
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # Seeds are transferred after the complete legacy seed list.
        return False

    @classmethod
    def transferred_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR, prior: list[Config]
    ) -> list[Config]:
        from ..cute.result_fission import analyze_result_fission
        from ..cute.result_fission_codegen import admitted_producer_keys
        from ..cute.result_fission_codegen import producer_keys_of

        spec = env.config_spec
        shape = spec.cute_result_fission_shape
        if not spec.cute_result_fission_available or shape is None:
            return []
        assert device_ir.host_function is not None
        with device_ir.host_function:
            proof = analyze_result_fission(env, device_ir, allow_unbound=True)
        if proof is None:
            return []
        candidates: list[dict[str, object]] = []
        for seed in prior:
            keys = producer_keys_of(seed)
            if keys is not None and keys not in candidates:
                candidates.append(keys)
        # Cap before admission: each admission compiles the producer once.
        admitted = admitted_producer_keys(proof, env, candidates[:MAX_PRODUCER_SEEDS])
        m, n, _ = shape
        consumer = default_consumer_block_sizes(m, n)
        layouts: list[tuple[list[int], list[int], int]] = [
            (list(consumer), list(consumer), 1)
        ]
        if n % 256 == 0 and m % 8 == 0:
            # An N-contiguous 4-wide twin (16-byte vectors, whole warps) for
            # every admitted producer; codegen refuses it unless the views
            # are cache-key aligned.
            layouts.append(([8, 256], [8, 64], 4))
        base = dict(spec._base_default_config().config)
        seeds: list[Config] = []
        for keys in admitted:
            for blocks, threads, vector in layouts:
                values = {
                    **base,
                    RESULT_FISSION_KEY: True,
                    **keys,
                    CONSUMER_BLOCK_SIZES_KEY: blocks,
                    CONSUMER_NUM_THREADS_KEY: threads,
                    CONSUMER_VECTOR_WIDTH_KEY: vector,
                }
                if "cute_pdl" in base:
                    values["cute_pdl"] = False
                candidate = Config.from_dict(values)
                try:
                    normalized = spec.normalized_config(candidate)
                except InvalidConfig:
                    continue
                if any(
                    normalized.config.get(key) != value
                    for key, value in values.items()
                    if key.startswith("cute_result_fission")
                ):
                    continue
                seeds.append(normalized)
        return seeds
