"""Typed full-tile cluster-K schedule for a proved private reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import exc
from ...runtime.config import Config
from .split_k_cluster_config import CLUSTER8_K4
from .split_k_cluster_config import SCHEDULE_KEY
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import ConfigValueExpression
    from ..device_ir import DeviceIR
    from .split_k_workspace import SplitKWorkspaceProof


@dataclass(frozen=True)
class ClusterKSchedule:
    """Physical geometry is an explicit schedule, independent of input extents."""

    bm: int = 16
    bn: int = 8
    bk: int = 128
    k_warps: int = 4
    cluster_ctas: int = 8

    @property
    def carrier_tile(self) -> tuple[int, int, int]:
        # Ordinary matmul block fields have an N16 minimum. The explicit
        # schedule owns physical N8; never widen the legacy N domain.
        return (self.bm, 16, self.bk)

    @property
    def shared_upper_bound(self) -> int:
        # Every allocation includes its maximum alignment gap. Reserve another
        # KiB for the CUDA cluster ABI (visible in the accepted native object).
        a = self.bm * self.bk * self.k_warps * 2
        b = self.bn * self.bk * self.k_warps * 2
        c = self.bm * self.bn * self.k_warps * 4
        return a + 1024 + b + 1024 + c + 128 + 1024


CLUSTER_K_SCHEDULE = ClusterKSchedule()


def _positive_config_leaves(
    expression: ConfigValueExpression | int | str, config: Config
) -> bool:
    if isinstance(expression, int):
        return True
    if isinstance(expression, str):
        value = config.config.get(expression)
        return type(value) is int and value > 0
    return all(_positive_config_leaves(arg, config) for arg in expression.arguments)


@dataclass(frozen=True)
class ClusterKFacts:
    m: int
    n: int
    k: int
    axes: tuple[int, int, int]
    chunk: ConfigValueExpression | int
    capacity_bytes: int

    def chunk_size(self, config: Config) -> int:
        from ..compile_environment import ConfigValueExpression

        return (
            self.chunk.evaluate(config)
            if isinstance(self.chunk, ConfigValueExpression)
            else self.chunk
        )

    def valid_config(self, spec: ConfigSpec, config: Config) -> bool:
        s = CLUSTER_K_SCHEDULE
        blocks = tuple(
            spec.block_sizes.config_get(config.block_sizes, axis) for axis in self.axes
        )
        threads = tuple(
            spec.num_threads.config_get(config.num_threads, axis) for axis in self.axes
        )
        return (
            blocks == s.carrier_tile
            and threads == (4, 8, 4)
            and _positive_config_leaves(self.chunk, config)
            and self.chunk_size(config) == self.k // s.cluster_ctas
            and self.capacity_bytes >= s.shared_upper_bound
            and config.get("cute_split_k_workspace", False) is False
            and config.get("cute_collective_mma", False) is False
        )


def cluster_facts(env: CompileEnvironment, ir: DeviceIR) -> ClusterKFacts | None:
    from .split_k_workspace import analyze_split_k_reduction

    proof = analyze_split_k_reduction(env, ir)
    if proof is None:
        return None
    s = CLUSTER_K_SCHEDULE
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device)
    if (
        proof.m % s.bm
        or proof.n % s.bn
        or proof.k % (s.cluster_ctas * s.k_warps * s.bk)
        # Copy addresses use nonnegative Int32 element offsets. Bounding the
        # dimensions alone does not cover padded row strides. The bias proof
        # already enforces the same bound; the fresh output is contiguous.
        or (proof.m - 1) * proof.lhs_strides[0] + proof.k - 1 >= 1 << 31
        or (proof.k - 1) * proof.rhs_strides[0] + proof.n - 1 >= 1 << 31
        or capacity < s.shared_upper_bound
        or (proof.m // s.bm) * (proof.n // s.bn) * s.cluster_ctas >= 1 << 31
    ):
        return None
    return ClusterKFacts(
        proof.m,
        proof.n,
        proof.k,
        (proof.m_block_id, proof.n_block_id, proof.k_block_id),
        proof.chunk,
        capacity,
    )


def validate_cluster_config(
    spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    if config.get(SCHEDULE_KEY) != CLUSTER8_K4:
        return
    facts = spec.cute_split_k_cluster_facts
    if facts is not None and facts.valid_config(spec, Config.from_dict(config)):
        return
    if fix_invalid:
        config.pop(SCHEDULE_KEY)
        return
    raise exc.InvalidConfig(
        "cluster8_k4 requires carrier M16/N16/K128, logical threads (4,8,4), "
        "eight full partitions and the complete shared-memory capacity"
    )


def checked_cluster_proof(
    env: CompileEnvironment, ir: DeviceIR, config: Config
) -> tuple[SplitKWorkspaceProof, ClusterKFacts]:
    from .split_k_workspace import analyze_split_k_reduction

    proof = analyze_split_k_reduction(env, ir)
    facts = cluster_facts(env, ir)
    if (
        proof is None
        or facts is None
        or not facts.valid_config(env.config_spec, config)
    ):
        raise exc.BackendUnsupported("cute", "cluster split-K proof declined")
    return proof, facts
