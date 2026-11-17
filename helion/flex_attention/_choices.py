"""Inductor's flex_attention choice hooks, answered with the Helion kernels.

Inductor's Triton lowering of flex_attention (and of its backward) asks the
active ``InductorChoices`` for extra candidates through
``append_flex_attention_choices`` / ``append_flex_attention_backward_choices``.
:class:`HelionFlexAttentionChoices` answers a call the Helion kernels run with
one choice, which replaces Inductor's Triton candidates; for any other call it
keeps them, and logs why at INFO (``HELION_LOGS=helion.flex_attention``).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.resources
import json
import logging
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
import uuid

import torch
from torch._environment import is_fbcode
from torch._inductor.choices import InductorChoices
from torch._inductor.ir import ChoiceCaller
from torch._inductor.lowering import lowerings
from torch._inductor.virtualized import V

from . import _kernels
from . import _lowering
from helion._compiler.backend_registry import get_backend
from helion.autotuner.base_cache import helion_key
from helion.runtime.settings import autotuning_disallowed_reason

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from importlib.abc import Traversable

    from torch._inductor.ir import Layout
    from torch._inductor.ir import TensorBox

log = logging.getLogger("helion.flex_attention")

aten = torch.ops.aten

# HELION_* environment variables that code generation does not read: those
# of the settings that change no generated code or that Helion writes back
# while it autotunes (and the LLM search's compile timeout), and those that
# only log or dump, check, locate or skip caches, drive the test harness or
# reach the LLM endpoint.  Any other variable, the ones code generation and
# the autotuner read directly included, keys the caches of generated code
# (``_keyed_env_var``), so new variables are keyed by default.
_NON_CODEGEN_ENV_VARS: frozenset[str] = frozenset(
    {
        "HELION_AOT_VERBOSE",
        "HELION_ASSERT_CACHE_HIT",
        "HELION_AUTOTUNER_TRITON_FAILURE_DUMP_DIR",
        "HELION_AUTOTUNE_BENCHMARK_SUBPROCESS",
        "HELION_AUTOTUNE_COMPILE_TIMEOUT",
        "HELION_AUTOTUNE_METRICS_JSON",
        "HELION_AUTOTUNE_PRECOMPILE",
        "HELION_AUTOTUNE_PRECOMPILE_JOBS",
        "HELION_AUTOTUNE_PROGRESS_BAR",
        "HELION_AUTOTUNE_RANDOM_SEED",
        "HELION_AUTOTUNE_SWEEP_RESULT_JSON",
        "HELION_CACHE_DIR",
        "HELION_CUTE_LAUNCH_SHAPE_CHECK",
        "HELION_CUTE_LAUNCH_SHAPE_LOG",
        "HELION_DEBUG_COLLAPSE",
        "HELION_DEBUG_DISTRIBUTED",
        "HELION_DELETE_CACHE_AFTER_TEST",
        "HELION_DISALLOW_AUTOTUNING",
        "HELION_DIST_CHECK_CONFIG_CONSISTANCY",
        "HELION_IGNORE_WARNINGS",
        "HELION_LLM_API_BASE",
        "HELION_LLM_API_KEY",
        "HELION_LLM_CA_BUNDLE",
        "HELION_LLM_CLIENT_CERT",
        "HELION_LLM_CLIENT_KEY",
        "HELION_LLM_COMPILE_TIMEOUT_S",
        "HELION_LLM_EXTRA_HEADERS",
        "HELION_LLM_VERTEX_LOCATION",
        "HELION_LLM_VERTEX_PROJECT",
        "HELION_LOGS",
        "HELION_MEASURE_COMPILE_TIME",
        "HELION_REMOTE_CACHE_BACKEND",
        "HELION_SKIP_CACHE",
    }
)
_NON_CODEGEN_ENV_PREFIXES: tuple[str, ...] = (
    "HELION_AUTOREVIEW_",
    "HELION_AUTOTUNE_LOG",
    "HELION_KEEP_",
    "HELION_PRINT_",
    "HELION_TEST",
)


def _keyed_env_var(name: str) -> bool:
    """Whether environment variable ``name`` keys the caches of the code
    Helion generates (see ``_NON_CODEGEN_ENV_VARS``)."""
    return (
        name.startswith("HELION_")
        and name not in _NON_CODEGEN_ENV_VARS
        and not name.startswith(_NON_CODEGEN_ENV_PREFIXES)
    )


def _package_source_hash(root: Traversable) -> str | None:
    """The hash of the ``.py`` files under ``root`` (their paths and
    contents), or None when there are none. ``root`` may be a directory or a
    package inside an archive (``importlib.resources.files``)."""
    digest = hashlib.sha256()
    found = False
    pending: list[tuple[str, Traversable]] = [("", root)]
    while pending:
        prefix, node = pending.pop()
        for child in sorted(node.iterdir(), key=lambda c: c.name):
            path = f"{prefix}{child.name}"
            if child.is_dir():
                pending.append((f"{path}/", child))
            elif child.name.endswith(".py"):
                found = True
                digest.update(path.encode())
                digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest() if found else None


@functools.cache
def _helion_source_key() -> str:
    """Helion's source hash: ``helion_key()``, which walks the source tree on
    disk. In fbcode the source may sit inside an archive that cannot walk,
    so it hashes the package's files through ``importlib.resources``, which
    reads archives too. Without source files to hash it is a key used only
    in this process: no graph compiled after the import is reused across
    processes, rather than one generated by another Helion."""
    if not is_fbcode():
        return helion_key()
    key = _package_source_hash(importlib.resources.files("helion"))
    if key is None:
        log.warning(
            "helion.flex_attention found no Helion source to key Inductor's "
            "FX graph cache on; graphs compiled after the import are not "
            "reused across processes"
        )
        return f"process-{uuid.uuid4().hex}"
    return key


@dataclasses.dataclass(frozen=True)
class Options:
    """What :func:`helion.flex_attention.configure` sets."""

    forward_config: Mapping[str, Any] | None = None
    backward_config: Mapping[str, Any] | None = None
    autotune_tolerance: float = _kernels.AUTOTUNE_TOLERANCE

    def config(self, direction: str) -> Mapping[str, Any] | None:
        """The configured config of the ``direction`` kernel, if any."""
        return self.forward_config if direction == "forward" else self.backward_config


options = Options()

# Whether the hooks answer (enable / disable). The contributor stays
# registered once imported: Inductor builds its choices handler once per
# thread, from the contributors registered at that time (the autograd thread,
# which compiles the backward, builds its own), so a contributor registered
# and unregistered at runtime would reach some threads and not others.
enabled = False


def cache_key() -> str:
    """What the code generated for a graph's flex_attention calls depends on
    beyond the graph and PyTorch: Helion's source (``_helion_source_key``),
    the CuTe DSL version, the ``HELION_*`` variables
    code generation reads, :data:`options`, and whether autotuning is
    disallowed (a kernel that would autotune then keeps Triton's)."""
    env = sorted((k, v) for k, v in os.environ.items() if _keyed_env_var(k))
    return json.dumps(
        [
            "helion.flex_attention",
            _helion_source_key(),
            get_backend("cute").codegen_versions(),
            env,
            [
                options.forward_config,
                options.backward_config,
                options.autotune_tolerance,
            ],
            autotuning_disallowed_reason() is not None,
        ],
        sort_keys=True,
    )


class _HelionChoice(ChoiceCaller):
    """The one choice for a flex_attention call the Helion kernels run.
    Inductor takes a lone choice without benchmarking it."""

    def __init__(
        self,
        name: str,
        input_nodes: list[Any],
        layout: Layout,
        output_node: Callable[[], TensorBox],
    ) -> None:
        super().__init__(
            name, input_nodes, layout, description="Helion CuTe flex_attention"
        )
        self._output_node = output_node

    def output_node(self) -> TensorBox:
        return self._output_node()

    def hash_key(self) -> str:
        return self.name

    def call_name(self) -> str:
        return self.name

    def to_callable(self) -> Callable[..., Any]:
        raise NotImplementedError(f"{self.name} is not benchmarked")

    def info_dict(self) -> dict[str, Any]:
        return {"backend": "helion"}


_Outputs = TypeVar("_Outputs")


def _plan(
    op: object, direction: str, plan: Callable[..., Callable[[], _Outputs]]
) -> Callable[[], _Outputs] | None:
    """The lowering of the call Inductor is lowering, or None (logged) when
    the Helion kernels do not run it. The hooks receive Inductor's lowered
    Triton inputs; the Helion kernels need the call's own arguments (its FX
    subgraphs and captured tensors), which the FX node being lowered maps to."""
    node = V.graph.current_node
    if node is None or node.target is not op:
        log.info(
            "flex_attention %s runs Inductor's kernel: lowered outside its FX node",
            direction,
        )
        return None
    args = torch.fx.node.map_arg(node.args, lambda n: V.graph.env[n])
    try:
        lower = plan(*args, options=options)
    except _lowering.Declined as e:
        log.info("flex_attention %s runs Inductor's kernel: %s", direction, e)
        return None
    return lower


def _write_into(dst: TensorBox, src: TensorBox) -> None:
    """Store ``src`` in ``dst``, a buffer Inductor's lowering allocated for its
    kernel to write and returns. This is a copy kernel (a choice returns one
    output, so the Helion kernel's other outputs cannot replace these
    buffers); it fuses with the producer's pointwise tail where there is one,
    such as grad_query's cast from float32."""
    lowerings[aten.copy_.default](dst, src)


class HelionFlexAttentionChoices(InductorChoices):
    def uuid(self) -> str:
        return cache_key() if enabled else "helion.flex_attention: disabled"

    def append_flex_attention_choices(
        self,
        choices: list[Any],
        configs: list[Any],
        input_nodes: list[Any],
        subgraphs: list[Any],
        layout: Layout,
        kernel_options: dict[str, Any],
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
    ) -> list[Any]:
        if not enabled:
            return choices
        lower = _plan(
            torch.ops.higher_order.flex_attention, "forward", _lowering.plan_forward
        )
        if lower is None:
            return choices
        # Inductor returns these buffers next to the output.
        logsumexp, max_scores = input_nodes[3], input_nodes[4]

        def output_node() -> TensorBox:
            out, lse, max_out = lower()
            _write_into(logsumexp, lse)
            if max_out is not None:
                _write_into(max_scores, max_out)
            return out

        return [
            _HelionChoice("helion_flex_attention", input_nodes, layout, output_node)
        ]

    def append_flex_attention_backward_choices(
        self,
        choices: list[Any],
        configs: list[Any],
        input_nodes: list[Any],
        subgraphs: list[Any],
        layout: Layout,
        kernel_options: dict[str, Any],
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
        *,
        mutated_inputs: list[Any],
    ) -> list[Any]:
        if not enabled:
            return choices
        lower = _plan(
            torch.ops.higher_order.flex_attention_backward,
            "backward",
            _lowering.plan_backward,
        )
        if lower is None:
            return choices
        # Inductor returns grad_query and grad_value from these buffers, and
        # grad_key from the choice's output.
        grad_query, grad_value = mutated_inputs[:2]

        def output_node() -> TensorBox:
            dq, dk, dv = lower()
            _write_into(grad_query, dq)
            _write_into(grad_value, dv)
            return dk

        return [
            _HelionChoice(
                "helion_flex_attention_backward", input_nodes, layout, output_node
            )
        ]
