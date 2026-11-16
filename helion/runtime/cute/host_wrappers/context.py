"""The host wrapper emitter protocol and the context an emitter appends to.

``_append_cute_wrapper_plan`` (``launcher``) builds one
:class:`HostWrapperContext` per wrapper plan -- the wrapper body lines and
kernel call arguments under construction, the plan, the device SM count,
the argument schema and the launch device -- and hands it to the
:class:`HostWrapperEmitter` registered for the plan's kind.  The context
carries the typed plan reads the emitters share; ``descriptors`` is the
wrapped grouped-descriptor proof a typed grouped RNA plan declares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .... import exc
from ...._compiler.cute.tcgen05_grouped_descriptors import WrappedGroupedDescriptorPlan

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import torch


def wrapped_grouped_descriptors(
    plan: dict[str, object],
) -> WrappedGroupedDescriptorPlan | None:
    """The plan's wrapped grouped-descriptor proof, re-checked against the
    typed flat provider it was derived from; None for plans without one."""
    descriptor_identity = plan.get("wrapped_grouped_descriptors")
    descriptors = (
        WrappedGroupedDescriptorPlan.from_identity(descriptor_identity)
        if descriptor_identity is not None
        else None
    )
    if descriptors is not None:
        if plan.get("orientation") != "nm":
            raise exc.BackendUnsupported(
                "cute", "wrapped descriptors require N,M orientation"
            )
        if plan.get("kind") in ("tcgen05_grouped_rna", "tcgen05_grouped_tma_rn"):
            expected_provider = (
                "typed_flat_element_bases_v1",
                descriptors.groups,
                (
                    descriptors.rows * descriptors.reduction,
                    descriptors.reduction,
                    descriptors.reduction,
                    16,
                ),
                (
                    descriptors.rows * descriptors.columns,
                    descriptors.columns,
                    descriptors.columns,
                    16,
                ),
                128,
                128,
                descriptors.offset_bits,
                (0, 1, 2, 3, "a_element_base", 5, 6, 7, "output_element_base"),
            )
            if (
                plan.get("flat_provider") != expected_provider
                or not plan.get("fixed_ab_tensormaps")
                or not plan.get("fixed_grouped_b_rank3")
                or plan.get("dynamic_ab_tensormaps")
                or plan.get("m_size") != descriptors.rows
                or plan.get("n_size") != descriptors.columns
                or plan.get("k_total_size") != descriptors.reduction
            ):
                raise exc.BackendUnsupported(
                    "cute", "wrapped descriptors disagree with typed provider"
                )
        elif plan.get("kind") == "tcgen05_d_tma":
            if (
                not plan.get("rank3_mnl_tensor")
                or plan.get("output_dtype") != "cutlass.Float32"
            ):
                raise exc.BackendUnsupported(
                    "cute", "wrapped output requires the flat FP32 store"
                )
        else:
            raise exc.BackendUnsupported(
                "cute", "wrapped descriptors require typed grouped RNA"
            )
    return descriptors


@dataclass
class HostWrapperContext:
    """What a host wrapper emitter appends to and reads."""

    body: list[str]
    call_args: list[str]
    plan: dict[str, object]
    num_sm: int | None
    schema_key: Sequence[tuple[object, ...]] | None
    device: torch.device | None
    descriptors: WrappedGroupedDescriptorPlan | None

    def plan_int(self, key: str, default: int | None = None) -> int:
        value = self.plan.get(key, default) if default is not None else self.plan[key]
        assert isinstance(value, int)
        return value

    def plan_optional_int(self, key: str) -> int | None:
        value = self.plan.get(key)
        assert value is None or isinstance(value, int)
        return value

    def plan_optional_str(self, key: str) -> str | None:
        value = self.plan.get(key)
        assert value is None or isinstance(value, str)
        return value

    def plan_optional_order(self, key: str) -> tuple[int, ...] | None:
        value = self.plan.get(key)
        if value is None:
            return None
        assert isinstance(value, (list, tuple))
        assert all(type(dim) is int for dim in value)
        order = tuple(int(dim) for dim in value)
        assert sorted(order) == list(range(len(order)))
        return order

    def append_permuted_cute_tensor_view(
        self,
        name: str,
        arg_idx: int,
        order: tuple[int, ...],
    ) -> None:
        shape = ", ".join(f"arg{arg_idx}_shape{dim}" for dim in order)
        stride = ", ".join(f"arg{arg_idx}_stride{dim}" for dim in order)
        self.body.append(
            f"    {name} = cute.make_tensor("
            f"arg{arg_idx}.iterator, "
            f"layout=cute.make_layout(({shape}), stride=({stride})))"
        )

    @staticmethod
    def require_positive_int(value: int | None, name: str) -> int:
        assert type(value) is int, name
        assert value > 0, name
        return value


@dataclass(frozen=True)
class HostWrapperEmitter:
    """The host half of one special-purpose emitter, keyed by its plan kind.

    ``emit`` appends the plan's wrapper body lines and kernel call arguments.
    ``namespace`` binds the host entries / tensor-map builders the body names
    into the ``@cute.jit`` wrapper's namespace (imported lazily: the producer
    modules pull the compiler in).  ``host_call`` spells the whole-root host
    entry call that replaces the generic kernel launch.  ``source_dependencies``
    are the package-relative helper sources the wrapper or kernel compiles
    that the generated text does not name, fingerprinted into the disk-cache
    key by ``source_dependencies.wrapper_source_dependencies``.
    """

    kind: str
    emit: Callable[[HostWrapperContext, dict[str, object]], None]
    namespace: Callable[[dict[str, object]], dict[str, object]] | None = None
    host_call: Callable[[list[str], dict[str, object]], None] | None = None
    source_dependencies: tuple[str, ...] = ()
