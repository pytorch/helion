from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import keyword
import math
import operator
import os
import pathlib
import tempfile
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from torch._functorch.aot_autograd import aot_module_simplified
import torch._functorch.config as functorch_config
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._higher_order_ops.scan import scan_op as _scan_op
from torch._inductor.decomposition import select_decomp_table
import torch.fx
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from .._compiler.device_ir import GraphInfo
    from .._compiler.host_function import HostFunction
    from ..runtime.kernel import Kernel


# Module-level lock to guard backward cache initialisation: without it, two
# threads can both see cache=None, create a new dict, and the second writer
# silently discards the first thread's results.  The lock also serialises
# AOT-autograd compilation, which has internal mutable state that is not
# thread-safe.
_backward_cache_lock: threading.Lock = threading.Lock()


@dataclass
class InputMapping:
    placeholder_name: str
    tensor_name: str
    fake_tensor: torch.Tensor | None


@dataclass
class InitSpec:
    """A carry init (`hl.zeros`/`hl.full`) lifted out of the FX graph as
    a graph input so the wrapper can construct it with
    ``requires_grad=True`` (mandatory for ``ScanAutogradOp`` to thread
    the carry-grad chain — see ``torch/_higher_order_ops/scan.py:450``).

    ``shape_dim_sources`` resolves each shape entry to either an
    ``(input_idx, dim)`` pair (use ``tensor_inputs[i].shape[d]``) or a
    concrete ``int``.
    """

    placeholder_name: str
    shape_dim_sources: list  # list[tuple[int, int] | int]
    fill_value: float
    dtype: torch.dtype
    device: torch.device | None


class GraphAnalyzer:
    """
    Analyzes forward Helion graph and extracts the pure computation subgraph.
    """

    def __init__(
        self,
        forward_graph: torch.fx.Graph,
        scalar_values: dict[str, object] | None = None,
        return_order: tuple[str, ...] | None = None,
        all_graphs: list | None = None,
        config_block_sizes: dict[int, int] | None = None,
        kernel_input_names: list[str] | None = None,
    ) -> None:
        self.forward_graph = forward_graph
        self.scalar_values = scalar_values or {}
        # Aligns compute-graph outputs with grad_outs (return-order).
        self.return_order = return_order
        self.all_graphs = all_graphs or []
        self.config_block_sizes = config_block_sizes or {}
        self.kernel_input_names = kernel_input_names or []
        self.init_specs: list[InitSpec] = []
        self.scan_combines: list[torch.fx.GraphModule] = []

    def _get_tensor_name(self, host_tensor_node: Node) -> str:
        target = host_tensor_node.target
        assert callable(target) and getattr(target, "__name__", "") == "_host_tensor"
        name = host_tensor_node.args[0]
        assert isinstance(name, str)
        return name

    def extract_computation_graph(
        self,
    ) -> tuple[
        torch.fx.Graph, list[InputMapping], list[tuple[int, ...]], list[tuple[int, ...]]
    ]:
        """
        Extract computation subgraph.

        Returns:
            compute_graph: Pure PyTorch FX graph
            input_mappings: Load -> placeholder mappings
            output_shapes: Shapes of each compute graph output (per-tile)
            host_output_shapes: Full host-tensor shapes of outputs
        """
        compute_graph = torch.fx.Graph()
        node_map: dict[Node, Node] = {}
        input_mappings: list[InputMapping] = []

        tensor_to_placeholder: dict[str, Node] = {}
        tensor_current_value: dict[str, Node] = {}
        host_output_shapes: dict[str, tuple[int, ...]] = {}

        for node in self.forward_graph.nodes:
            if node.op != "call_function":
                continue

            target = node.target
            assert callable(target)
            target_name = target.__name__
            target_module = getattr(target, "__module__", "") or ""

            # Lift carry inits as graph inputs: ScanAutogradOp zeros the
            # carry-grad chain unless `init.requires_grad=True`, which
            # `aten.full` doesn't accept.
            if target_module.startswith("helion.language") and target_name == "full":
                self._emit_init_placeholder(node, compute_graph, node_map)
                continue

            # Also detect aten.zeros_like / aten.full_like / aten.full used
            # as carry inits (e.g. ``torch.zeros_like(mean)``).
            if target in (
                torch.ops.aten.zeros_like.default,
                torch.ops.aten.full_like.default,
                torch.ops.aten.full.default,
            ):
                self._emit_aten_init_placeholder(node, compute_graph, node_map)
                continue

            if target_name in ("_for_loop", "_for_loop_step"):
                self._emit_scan_for_loop(
                    node,
                    compute_graph,
                    node_map,
                    input_mappings,
                    tensor_to_placeholder,
                    tensor_current_value,
                )
                continue

            if target_name == "_associative_scan":
                self._emit_associative_scan(node, compute_graph, node_map)
                continue

            # The scan emitter pre-binds `_phi` nodes; otherwise alias
            # `args[1]` (the post-loop value).
            if target_name == "_phi":
                if node in node_map:
                    continue
                final = node.args[1]
                if isinstance(final, Node) and final in node_map:
                    node_map[node] = node_map[final]
                continue

            # `getitem` on an unmapped (rewritten-out) `_for_loop` is dead.
            if target_name == "getitem":
                container = node.args[0]
                if isinstance(container, Node) and container not in node_map:
                    continue

            if target_name == "_new_var":
                src = node.args[0]
                if isinstance(src, Node) and src in node_map:
                    node_map[node] = node_map[src]
                continue

            if target_name == "load":
                host_tensor_node = node.args[0]
                assert isinstance(host_tensor_node, Node)
                tensor_name = self._get_tensor_name(host_tensor_node)
                fake_tensor = node.meta["val"]

                if tensor_name in tensor_current_value:
                    stored_value_node = tensor_current_value[tensor_name]
                    node_map[node] = node_map[stored_value_node]
                elif tensor_name in tensor_to_placeholder:
                    node_map[node] = tensor_to_placeholder[tensor_name]
                else:
                    ph_name = f"tile_{tensor_name}"
                    ph = compute_graph.placeholder(ph_name)
                    tensor_to_placeholder[tensor_name] = ph
                    node_map[node] = ph

                    input_mappings.append(
                        InputMapping(
                            placeholder_name=ph.name,
                            tensor_name=tensor_name,
                            fake_tensor=fake_tensor,
                        )
                    )

            elif target_name == "store":
                host_tensor_node = node.args[0]
                assert isinstance(host_tensor_node, Node)
                tensor_name = self._get_tensor_name(host_tensor_node)
                value_node = node.args[2]
                if isinstance(value_node, Node):
                    tensor_current_value[tensor_name] = value_node
                # Track host tensor shape for output reshape canonicalization.
                host_fake = host_tensor_node.meta.get("val")
                if isinstance(host_fake, torch.Tensor):
                    host_output_shapes[tensor_name] = tuple(
                        int(s) for s in host_fake.shape
                    )

            elif target_name == "_inductor_lowering_extra":
                data_inputs = node.args[0]
                assert isinstance(data_inputs, (list, tuple)) and len(data_inputs) >= 1
                assert isinstance(data_inputs[0], Node)
                node_map[node] = node_map[data_inputs[0]]

            elif target_name == "_mask_to":
                assert isinstance(node.args[0], Node)
                node_map[node] = node_map[node.args[0]]

            elif target_name == "_get_symnode":
                # User scalars (e.g. `eps`) get mapped; Helion-internal
                # symnodes (e.g. block sizes) stay unmapped — `_assert_all_mapped`
                # below catches any that flow into a real op.
                sym_name = node.args[0]
                assert isinstance(sym_name, str)
                val = _resolve_symnode_expr(sym_name, self.scalar_values)
                # Fallback: resolve block_size_X names via config_block_sizes.
                # These are Helion-internal symnodes for tile block sizes and
                # are not present in scalar_values (which only has user scalars).
                if val is None and sym_name.startswith("block_size_"):
                    try:
                        bid = int(sym_name[len("block_size_") :])
                        val = self.config_block_sizes.get(bid)
                    except ValueError:
                        pass
                if val is not None:
                    const_node = compute_graph.call_function(
                        torch.ops.aten.scalar_tensor.default,
                        (val,),  # pyrefly: ignore [bad-argument-type]
                    )
                    node_map[node] = const_node

            elif target_name == "subscript":
                self._lower_subscript(node, compute_graph, node_map)

            elif target_name == "dot":
                # Lower hl.dot -> a rank-appropriate aten matmul, mirroring the
                # scan-path lowering. Crucially this consumes only the two operand
                # args, so hl.dot's `acc=None`/`out_dtype=None` defaults never
                # reach the None-coalescing fixup below (which would otherwise
                # replace `out_dtype=None` with a tensor and break AOT diff).
                a0, a1 = node.args[0], node.args[1]
                assert isinstance(a0, Node) and isinstance(a1, Node)
                lhs = node_map[a0]
                rhs = node_map[a1]

                def _node_dtype(operand: Node) -> torch.dtype | None:
                    v = operand.meta.get("val") if operand.meta else None
                    return v.dtype if isinstance(v, torch.Tensor) else None

                ldt, rdt = _node_dtype(a0), _node_dtype(a1)
                if ldt is not None and rdt is not None and ldt != rdt:
                    # aten matmul rejects mixed-dtype operands (unlike tl.dot).
                    target_dt = torch.promote_types(ldt, rdt)
                    if ldt != target_dt:
                        lhs = compute_graph.call_function(
                            torch.ops.aten._to_copy.default,
                            (lhs,),
                            {"dtype": target_dt},
                        )
                    if rdt != target_dt:
                        rhs = compute_graph.call_function(
                            torch.ops.aten._to_copy.default,
                            (rhs,),
                            {"dtype": target_dt},
                        )
                mm_op = self._matmul_op_for_ranks(a0, a1)
                dot_node = compute_graph.call_function(
                    mm_op,  # pyrefly: ignore [bad-argument-type]
                    (lhs, rhs),
                )
                acc = node.args[2] if len(node.args) > 2 else node.kwargs.get("acc")
                if isinstance(acc, Node):
                    # hl.dot(a, b, acc=c) == c + a @ b
                    new_node = compute_graph.call_function(
                        torch.ops.aten.add.Tensor, (node_map[acc], dot_node)
                    )
                else:
                    new_node = dot_node
                if node.meta:
                    new_node.meta = node.meta.copy()
                node_map[node] = new_node

            elif target_name != "_host_tensor":
                # Restore None placeholders from strip_unused_inputs (duplicate
                # arg coalescing) and reinject `_extra_args` kwargs.
                # Inject extra_args FIRST: the first_node_arg fill consumes all
                # Nones and leaves no slots for extra_args injection.  Order:
                # 1. fill known extra_args into their specific None slots
                # 2. fill any remaining Nones with first_node_arg (strip_unused
                #    coalescing always uses the first non-None node arg as a
                #    placeholder for the duplicated arg)
                args = node.args
                kwargs = dict(node.kwargs)
                extra_args = kwargs.pop("_extra_args", None)
                if extra_args is not None and isinstance(extra_args, (list, tuple)):
                    args_list = list(args)
                    extra_idx = 0
                    for i, a in enumerate(args_list):
                        if a is None and extra_idx < len(extra_args):
                            args_list[i] = (  # pyrefly: ignore [unsupported-operation]
                                extra_args[extra_idx]
                            )
                            extra_idx += 1
                    args = tuple(args_list)

                # After extra_args injection, fill any remaining Nones caused
                # by strip_unused_inputs coalescing with first_node_arg.
                first_node_arg = next((a for a in args if isinstance(a, Node)), None)
                if first_node_arg is not None:
                    args = tuple(first_node_arg if a is None else a for a in args)

                _assert_all_mapped(node, args, kwargs, node_map)
                new_args = map_arg(args, node_map.get)
                new_kwargs = map_arg(kwargs, node_map.get)
                target = node.target
                assert callable(target)
                new_node = compute_graph.call_function(target, new_args, new_kwargs)
                if node.meta:
                    new_node.meta = node.meta.copy()
                node_map[node] = new_node

        input_tensor_names = set(tensor_to_placeholder.keys())
        stored_outputs = {
            t: v for t, v in tensor_current_value.items() if t not in input_tensor_names
        }

        # Output order = kernel return order; unnamed tensors keep store order.
        # When return_order is available, only include tensors that are
        # actually returned — intermediate buffers (e.g. `temp`) must be
        # excluded so the compute_graph output count matches grad_outs.
        if self.return_order is not None:
            return_set = set(self.return_order)
            # Also match stores to views of returned tensors (e.g.,
            # `out_flat = out.view(-1); out_flat[tile] = result; return out`).
            filtered: dict[str, Node] = {}
            for t, v in stored_outputs.items():
                if t in return_set:
                    filtered[t] = v
                else:
                    # Check if store name is a derived form of a return name.
                    for rname in return_set:
                        if t.startswith((rname + "_", rname + ".")):
                            filtered[rname] = v
                            break
            stored_outputs = filtered
            ordered_names: list[str] = []
            seen: set[str] = set()
            for name in self.return_order:
                if name in stored_outputs and name not in seen:
                    ordered_names.append(name)
                    seen.add(name)
        else:
            ordered_names = list(stored_outputs.keys())

        output_value_nodes = [stored_outputs[n] for n in ordered_names]
        outputs = [node_map[v] for v in output_value_nodes]
        compute_graph.output(tuple(outputs))

        output_shapes: list[tuple[int, ...]] = []
        host_shapes: list[tuple[int, ...]] = []
        for name, v in zip(ordered_names, output_value_nodes, strict=True):
            assert "val" in v.meta, (
                f"compute graph output {v.name} has no fake-tensor meta"
            )
            fake = v.meta["val"]
            output_shapes.append(tuple(int(s) for s in fake.shape))
            host_shapes.append(
                host_output_shapes.get(name, tuple(int(s) for s in fake.shape))
            )
        return compute_graph, input_mappings, output_shapes, host_shapes

    @staticmethod
    def _lower_subscript(
        node: Node,
        target_graph: torch.fx.Graph,
        node_map: dict[Node, Node],
    ) -> None:
        """Translate ``helion_language_view_ops_subscript(t, [slices])``
        into ``aten.unsqueeze.default`` (when one ``None`` for an
        unsqueeze) or pass-through (zero-None). Multi-None or non-slice
        indices raise.

        Used by the root walker AND the combine_fn builder so subscript
        ops never leak into AOT autograd's retrace (Helion ops require a
        compile environment that AOT doesn't have).
        """
        tensor_node = node.args[0]
        assert isinstance(tensor_node, Node)
        index = node.args[1]
        assert isinstance(index, (list, tuple))
        index_seq = list(index)
        none_pos = [i for i, idx in enumerate(index_seq) if idx is None]
        non_slice = [
            i
            for i, idx in enumerate(index_seq)
            if idx is not None and idx != slice(None)
        ]
        if non_slice:
            raise exc.AutodiffNotSupported(
                f"subscript with non-slice index: {index_seq}"
            )
        if len(none_pos) == 1:
            new_node = target_graph.call_function(
                torch.ops.aten.unsqueeze.default,
                (node_map[tensor_node], none_pos[0]),
            )
            if node.meta:
                new_node.meta = node.meta.copy()
            node_map[node] = new_node
        elif len(none_pos) == 0:
            node_map[node] = node_map[tensor_node]
        else:
            raise exc.AutodiffNotSupported(f"subscript with multiple None: {index_seq}")

    def _emit_init_placeholder(
        self,
        node: Node,
        compute_graph: torch.fx.Graph,
        node_map: dict[Node, Node],
    ) -> None:
        """Lift an `hl.zeros`/`hl.full` carry init to a compute_graph
        placeholder. Records an InitSpec so the wrapper can construct
        the tensor with `requires_grad=True` at AOT call time.
        """
        shape_arg = node.args[0]
        fill_value = node.args[1]
        dtype = node.args[2]
        assert isinstance(shape_arg, (list, tuple))
        assert isinstance(fill_value, (int, float))
        assert isinstance(dtype, torch.dtype)
        shape_dim_sources: list[tuple[int, int] | int] = []
        for s in shape_arg:
            if isinstance(s, Node):
                shape_dim_sources.append(self._resolve_block_size_to_input_dim(s))
            elif isinstance(s, int):
                shape_dim_sources.append(s)
            else:
                raise exc.AutodiffNotSupported(
                    f"unsupported shape entry in hl.full: {s!r}"
                )
        idx = len(self.init_specs)
        ph_name = f"init_{idx}"
        ph = compute_graph.placeholder(ph_name)
        node_map[node] = ph
        fake = node.meta.get("val")
        device = fake.device if isinstance(fake, torch.Tensor) else None
        self.init_specs.append(
            InitSpec(
                # Use ph.name (actual FX-assigned name), not ph_name
                # (requested name): FX deduplicates placeholder names so
                # ph.name may differ from ph_name after deduplication.
                placeholder_name=ph.name,
                shape_dim_sources=shape_dim_sources,
                fill_value=float(fill_value),
                dtype=dtype,
                device=device,
            )
        )

    def _emit_aten_init_placeholder(
        self,
        node: Node,
        compute_graph: torch.fx.Graph,
        node_map: dict[Node, Node],
    ) -> None:
        """Lift an ``aten.zeros_like`` / ``aten.full_like`` / ``aten.full``
        carry init to a compute_graph placeholder. The shape is taken from
        the fake-tensor metadata (concrete ints), so this works even when
        the aten op references block-size symnodes that can't be resolved
        through the ``_get_symnode`` path.
        """
        fake = node.meta.get("val")
        if not isinstance(fake, torch.Tensor):
            raise exc.AutodiffNotSupported(
                f"aten init op {node.target!r} has no fake-tensor meta"
            )
        target = node.target
        if (
            target is torch.ops.aten.full_like.default
            or target is torch.ops.aten.full.default
        ):
            fill_value = float(node.args[1]) if len(node.args) >= 2 else 0.0  # pyrefly: ignore [bad-argument-type]
        else:
            fill_value = 0.0
        dtype = fake.dtype
        # Extract dtype override from kwargs if present.
        if "dtype" in node.kwargs and isinstance(node.kwargs["dtype"], torch.dtype):
            dtype = node.kwargs["dtype"]
        # Try to resolve shape from args (symnode list) first — this gives
        # correct input-dim mappings. Fall back to fake.shape concrete ints
        # only if the args don't contain resolvable symnodes.
        shape_dim_sources: list[tuple[int, int] | int] = []
        shape_arg = node.args[0] if node.args else None
        if isinstance(shape_arg, (list, tuple)):
            for s in shape_arg:
                if isinstance(s, Node):
                    try:
                        shape_dim_sources.append(
                            self._resolve_block_size_to_input_dim(s)
                        )
                        continue
                    except exc.AutodiffNotSupported:
                        pass
                if isinstance(s, int):
                    shape_dim_sources.append(s)
                elif isinstance(s, Node):
                    sf = s.meta.get("val")
                    shape_dim_sources.append(int(sf) if sf is not None else int(s))  # pyrefly: ignore [bad-argument-type]
                else:
                    shape_dim_sources.append(int(s))  # pyrefly: ignore [bad-argument-type]
        else:
            shape_dim_sources = [int(s) for s in fake.shape]
        idx = len(self.init_specs)
        ph_name = f"init_{idx}"
        ph = compute_graph.placeholder(ph_name)
        node_map[node] = ph
        self.init_specs.append(
            InitSpec(
                # Use ph.name (actual FX-assigned name), not ph_name
                # (requested name): FX deduplicates placeholder names so
                # ph.name may differ from ph_name after deduplication.
                placeholder_name=ph.name,
                shape_dim_sources=shape_dim_sources,
                fill_value=fill_value,
                dtype=dtype,
                device=fake.device,
            )
        )

    def _resolve_block_size_to_input_dim(self, sym_node: Node) -> tuple[int, int] | int:
        """For a `_get_symnode('block_size_X')` node, return either
        ``(input_idx, dim)`` (so the wrapper uses
        ``tensor_inputs[input_idx].shape[dim]``) or a concrete ``int``.

        Two-pass:
          1. Direct: block_size_X appears in some load/store subscript
             at dim ``d`` of host tensor ``T``. If T is a kernel input,
             return ``(input_names.index(T), d)``. If T is an output,
             return the concrete int from T's fake-tensor meta.
          2. Indirect via carry chain: `hl.full([block_size_X, …])` is
             outer_args[i] of some `_for_loop`; the i-th subgraph
             placeholder's `sym_size_int(., d)` is used as a load/store
             subscript → pin `block_size_X` to the host's load dim.
        """
        if (
            sym_node.op != "call_function"
            or getattr(sym_node.target, "__name__", "") != "_get_symnode"
        ):
            raise exc.AutodiffNotSupported(
                f"non-symnode shape arg: {sym_node.target!r}"
            )
        bs_name = sym_node.args[0]
        input_names = list(self.kernel_input_names)

        # Pass 1: direct.
        for gi in self.all_graphs:
            for n in gi.graph.nodes:
                if n.op != "call_function":
                    continue
                if getattr(n.target, "__name__", "") not in ("load", "store"):
                    continue
                sub = n.args[1]
                if not isinstance(sub, (list, tuple)):
                    continue
                host = n.args[0]
                if not isinstance(host, Node):
                    continue
                tname = host.args[0]
                assert isinstance(tname, str)
                for d, idx in enumerate(sub):
                    if (
                        isinstance(idx, Node)
                        and getattr(idx.target, "__name__", "") == "_get_symnode"
                        and idx.args[0] == bs_name
                    ):
                        if tname in input_names:
                            return (input_names.index(tname), d)
                        host_fake = host.meta.get("val")
                        assert isinstance(host_fake, torch.Tensor)
                        return int(host_fake.shape[d])

        # Pass 2: carry chain. `hl.full([block_size_X, …])` flows into
        # `_for_loop`'s outer_args; the matching subgraph placeholder's
        # `sym_size_int` is consumed as a load/store subscript.
        from .._compiler.device_ir import RootGraphInfo

        for gi in self.all_graphs:
            if not isinstance(gi, RootGraphInfo):
                continue
            for n in gi.graph.nodes:
                if (
                    n.op != "call_function"
                    or getattr(n.target, "__name__", "") != "_for_loop"
                ):
                    continue
                outer = n.args[3]
                sub_graph_id = n.args[0]
                assert isinstance(outer, (list, tuple))
                assert isinstance(sub_graph_id, int)
                sub_info = self.all_graphs[sub_graph_id]
                sub_phs = list(sub_info.graph.find_nodes(op="placeholder"))
                for arg_i, outer_arg in enumerate(outer):
                    if not isinstance(outer_arg, Node):
                        continue
                    target_module = getattr(outer_arg.target, "__module__", "") or ""
                    if not (
                        target_module.startswith("helion.language")
                        and getattr(outer_arg.target, "__name__", "") == "full"
                    ):
                        continue
                    shape_arg = outer_arg.args[0]
                    assert isinstance(shape_arg, (list, tuple))
                    dim_to_bs: dict[int, str] = {}
                    for d, s in enumerate(shape_arg):
                        if (
                            isinstance(s, Node)
                            and getattr(s.target, "__name__", "") == "_get_symnode"
                        ):
                            sym_name_arg = s.args[0]
                            assert isinstance(sym_name_arg, str)
                            dim_to_bs[d] = sym_name_arg
                    placeholder = sub_phs[arg_i]
                    for sub_n in sub_info.graph.nodes:
                        if sub_n.op != "call_function":
                            continue
                        if sub_n.target is not torch.ops.aten.sym_size.int:
                            continue
                        if sub_n.args[0] is not placeholder:
                            continue
                        d = sub_n.args[1]
                        if dim_to_bs.get(d) != bs_name:
                            continue
                        for use in sub_n.users:
                            if getattr(use.target, "__name__", "") not in (
                                "load",
                                "store",
                            ):
                                continue
                            sub_use = use.args[1]
                            if not isinstance(sub_use, (list, tuple)):
                                continue
                            for use_d, idx in enumerate(sub_use):
                                if idx is sub_n:
                                    host = use.args[0]
                                    assert isinstance(host, Node)
                                    tname = host.args[0]
                                    assert isinstance(tname, str)
                                    if tname in input_names:
                                        return (
                                            input_names.index(tname),
                                            use_d,
                                        )
                                    host_fake = host.meta.get("val")
                                    assert isinstance(host_fake, torch.Tensor)
                                    return int(host_fake.shape[use_d])
        raise exc.AutodiffNotSupported(f"cannot resolve {bs_name!r} to an input dim")

    def _emit_associative_scan(
        self,
        node: Node,
        compute_graph: torch.fx.Graph,
        node_map: dict[Node, Node],
    ) -> None:
        """Lower a Helion ``_associative_scan`` (cumulative scan) into a
        differentiable aten op so AOT autograd can handle it.

        Args are ``(combine_graph_id, input, dim, reverse, ...)``. When the
        combine helper is a single ``add`` it is a cumulative sum →
        ``aten.cumsum`` (whose backward is a reverse cumsum that Helion can
        codegen via ``flip``+``cumsum``). Other combiners (``mul``=cumprod,
        ``maximum``=cummax, …) and reverse scans are not yet supported.
        """
        graph_id = node.args[0]
        input_node = node.args[1]
        dim = node.args[2]
        reverse = bool(node.args[3]) if len(node.args) > 3 else False
        assert isinstance(graph_id, int)
        assert isinstance(input_node, Node)
        assert isinstance(dim, int)
        if input_node not in node_map:
            raise exc.AutodiffNotSupported(
                f"associative_scan input {input_node.name} not in node_map"
            )
        if reverse:
            raise exc.AutodiffNotSupported("reverse associative_scan not supported")
        helper = self.all_graphs[graph_id]
        combine_ops = [
            getattr(n.target, "_opname", None) or getattr(n.target, "__name__", "")
            for n in helper.graph.nodes
            if n.op == "call_function"
        ]
        if combine_ops != ["add"]:
            raise exc.AutodiffNotSupported(
                f"associative_scan with combine {combine_ops} is not supported "
                "(only cumulative sum / add is differentiable here)"
            )
        new_node = compute_graph.call_function(
            torch.ops.aten.cumsum.default, (node_map[input_node], dim)
        )
        if node.meta:
            new_node.meta = node.meta.copy()
        node_map[node] = new_node

    def _emit_scan_for_loop(
        self,
        for_loop_node: Node,
        compute_graph: torch.fx.Graph,
        node_map: dict[Node, Node],
        input_mappings: list,
        tensor_to_placeholder: dict[str, Node],
        tensor_current_value: dict[str, Node],
    ) -> None:
        """Replace one ``_for_loop(graph_id, …)`` with a ``scan_op`` call.

        Carries (outer_args with ``_phi(arg, getitem)`` users) become the
        scan's ``init``. TILED loads (subscript uses the inner block size)
        become reshaped scan-form ``xs``; broadcast loads and non-carry
        outer args become ``additional_inputs``. TILED stores become
        ``ys`` (un-reshaped after the scan).
        """
        graph_id = for_loop_node.args[0]
        outer_args_raw = for_loop_node.args[3]
        assert isinstance(outer_args_raw, (list, tuple))
        outer_args = list(outer_args_raw)
        assert isinstance(graph_id, int)
        sub_info = self.all_graphs[graph_id]
        sub_graph = sub_info.graph
        block_ids = sub_info.block_ids
        if len(block_ids) != 1:
            raise exc.AutodiffNotSupported(
                f"scan rewrite needs single-block tile loops, got {block_ids}"
            )
        block_id = block_ids[0]
        if block_id not in self.config_block_sizes:
            raise exc.AutodiffNotSupported(
                f"scan loop block_id={block_id} not present in configured "
                f"block_sizes {sorted(self.config_block_sizes)}"
            )
        block_size = self.config_block_sizes[block_id]
        bs_name = f"block_size_{block_id}"

        # Detect carries: an outer_arg is a carry iff it has a _phi user.
        # Walk outer_args directly (not getitem indices) because non-carry
        # args (e.g., attention's pre-loop `q`) can appear at any position.
        carry_indices: list[int] = []
        for arg_i, oa in enumerate(outer_args):
            if not isinstance(oa, Node):
                continue
            if any(
                getattr(u.target, "__name__", "") == "_phi" and u.args[0] is oa
                for u in oa.users
            ):
                carry_indices.append(arg_i)
        additional_indices = [
            i for i in range(len(outer_args)) if i not in carry_indices
        ]

        # Build bs_equiv_nodes for sym_size_int-based subscripts.
        sub_phs = list(sub_graph.find_nodes(op="placeholder"))
        bs_equiv_nodes: set[Node] = set()
        for ci in carry_indices:
            outer_arg = outer_args[ci]
            if not isinstance(outer_arg, Node):
                continue
            oa_module = getattr(outer_arg.target, "__module__", "") or ""
            if not (
                oa_module.startswith("helion.language")
                and getattr(outer_arg.target, "__name__", "") == "full"
            ):
                continue
            shape_arg = outer_arg.args[0]
            if not isinstance(shape_arg, (list, tuple)):
                continue
            bs_dims: set[int] = set()
            for d, s in enumerate(shape_arg):
                if (
                    isinstance(s, Node)
                    and getattr(s.target, "__name__", "") == "_get_symnode"
                    and s.args[0] == bs_name
                ):
                    bs_dims.add(d)
            if not bs_dims:
                continue
            ph = sub_phs[ci]
            for sn in sub_graph.nodes:
                if (
                    sn.op == "call_function"
                    and sn.target is torch.ops.aten.sym_size.int
                    and sn.args[0] is ph
                    and sn.args[1] in bs_dims
                ):
                    bs_equiv_nodes.add(sn)

        tiled_loads: list = []
        bcast_loads: list = []
        stores: list = []
        for n in sub_graph.nodes:
            if n.op != "call_function":
                continue
            tn = getattr(n.target, "__name__", "")
            if tn == "load":
                host = n.args[0]
                tname = host.args[0]
                td = self._subscript_tile_dim(n.args[1], bs_name, bs_equiv_nodes)
                if td is not None:
                    tiled_loads.append((n, tname, td))
                else:
                    bcast_loads.append((n, tname))
            elif tn == "store":
                host = n.args[0]
                tname = host.args[0]
                td = self._subscript_tile_dim(n.args[1], bs_name, bs_equiv_nodes)
                if td is None:
                    # Detect "point stores" that use a tile_id integer index as
                    # the scan-position subscript (e.g. h[i_b, t_i.id, …]).
                    # These write one result per scan step at an integer chunk
                    # index rather than using the tiled subscript form.
                    # We cannot map them to scan_op ys (which require a
                    # consistent tensor slice per step), so raise a specific
                    # AutodiffNotSupported rather than crashing later with an
                    # uninformative ValueError.
                    if self._subscript_has_tile_id(n.args[1]):
                        raise exc.AutodiffNotSupported(
                            f"scan loop store to {tname!r} uses a tile_id "
                            "(integer chunk index) subscript, which cannot be "
                            "expressed as a scan_op ys output; autodiff for "
                            "this store pattern is not yet supported"
                        )
                    raise exc.AutodiffNotSupported(
                        f"AGGREGATE_STORE in tile loop (store to {tname!r}) "
                        "doesn't use the inner tile var"
                    )
                stores.append((n, tname, td, n.args[2]))

        init_nodes: list[Node] = []
        for ci in carry_indices:
            outer_arg = outer_args[ci]
            assert isinstance(outer_arg, Node)
            if outer_arg not in node_map:
                raise exc.AutodiffNotSupported(
                    f"carry init {outer_arg.name} not in node_map"
                )
            init_nodes.append(node_map[outer_arg])

        # Fix carry init shapes: when a carry's shape includes the scan's
        # block_size dimension, the InitSpec incorrectly resolves it to the
        # full tensor dimension (e.g. V=64) instead of the per-step block
        # size (e.g. 32).  Patch the InitSpec so differentiate_graph creates
        # the init tensor with the correct per-step shape.
        for ci in carry_indices:
            outer_arg = outer_args[ci]
            if not isinstance(outer_arg, Node):
                continue
            oa_module = getattr(outer_arg.target, "__module__", "") or ""
            oa_name = getattr(outer_arg.target, "__name__", "")
            is_hl_full = oa_module.startswith("helion.language") and oa_name == "full"
            is_aten_init = outer_arg.target in (
                torch.ops.aten.zeros_like.default,
                torch.ops.aten.full_like.default,
                torch.ops.aten.full.default,
            )
            if not (is_hl_full or is_aten_init):
                continue
            init_ph = node_map[outer_arg]
            # Find the matching InitSpec by placeholder name.
            matching_spec = None
            for spec in self.init_specs:
                if spec.placeholder_name == init_ph.name:
                    matching_spec = spec
                    break
            if matching_spec is None:
                continue
            if is_hl_full:
                shape_arg = outer_arg.args[0]
                if not isinstance(shape_arg, (list, tuple)):
                    continue
                for d, s in enumerate(shape_arg):
                    if not (
                        isinstance(s, Node)
                        and getattr(s.target, "__name__", "") == "_get_symnode"
                    ):
                        continue
                    sym_name = s.args[0]
                    if sym_name == bs_name:
                        # This dimension uses the scan's block_size; replace
                        # the resolved (input_idx, dim) with the concrete
                        # block_size so the init tensor matches the per-step
                        # carry shape.
                        if d < len(matching_spec.shape_dim_sources):
                            matching_spec.shape_dim_sources[d] = block_size
                    elif isinstance(sym_name, str) and sym_name.startswith(
                        "block_size_"
                    ):
                        # This dimension uses an outer GRID tile block_size.
                        # _resolve_block_size_to_input_dim returns the FULL
                        # tensor dimension (e.g. BT=32) which is correct for
                        # the full-batch scan context — each scan step
                        # operates on ALL grid cells simultaneously, so the
                        # init carry must have the full outer dimension, not
                        # the per-cell block_size.  Keep the resolved
                        # (input_idx, dim) as-is; it already encodes the full
                        # dimension and will be correctly read at
                        # init-tensor-construction time.
                        #
                        # However, if the resolved source is already a
                        # concrete int that differs from the full tensor dim
                        # (e.g. it was incorrectly patched to the per-cell
                        # block_size by a previous pass), restore it from
                        # _resolve_block_size_to_input_dim.
                        if d < len(matching_spec.shape_dim_sources) and isinstance(
                            matching_spec.shape_dim_sources[d], int
                        ):
                            try:
                                resolved_src = self._resolve_block_size_to_input_dim(s)
                                matching_spec.shape_dim_sources[d] = resolved_src
                            except exc.AutodiffNotSupported:
                                pass
            elif is_aten_init:
                # For aten init ops, the shape comes from the fake tensor.
                # Only replace the dimension that corresponds to the tiled
                # (scan) dimension.  Identify it via sym_size_int calls on
                # the carry's subgraph placeholder that appear in load/store
                # subscripts at the tile-dim position.
                ph = sub_phs[ci]
                tile_dims: set[int] = set()
                for sn in sub_graph.nodes:
                    if (
                        sn.op == "call_function"
                        and sn.target is torch.ops.aten.sym_size.int
                        and sn.args[0] is ph
                    ):
                        sym_dim = sn.args[1]
                        for use in sn.users:
                            use_name = getattr(use.target, "__name__", "")
                            if use_name not in ("load", "store"):
                                continue
                            sub_subscript = use.args[1]
                            if not isinstance(sub_subscript, (list, tuple)):
                                continue
                            td = self._subscript_tile_dim(
                                sub_subscript, bs_name, bs_equiv_nodes
                            )
                            if td is not None:
                                for idx in sub_subscript:
                                    if idx is sn:
                                        tile_dims.add(sym_dim)
                for d in tile_dims:
                    if d < len(matching_spec.shape_dim_sources):
                        matching_spec.shape_dim_sources[d] = block_size

        xs_nodes: list[Node] = []
        # Track padded dim sizes per (tname, tile_dim) for use in stores unreshape.
        tiled_load_padded: dict[tuple[str, int], int] = {}
        for ln, tname, td in tiled_loads:
            host_node = ln.args[0]
            if tname in tensor_to_placeholder:
                ph = tensor_to_placeholder[tname]
            else:
                host_fake = host_node.meta["val"]
                ph_name = f"tile_{tname}"
                ph = compute_graph.placeholder(ph_name)
                tensor_to_placeholder[tname] = ph
                input_mappings.append(
                    InputMapping(
                        placeholder_name=ph.name,
                        tensor_name=tname,
                        fake_tensor=host_fake,
                    )
                )
            host_fake = host_node.meta["val"]
            host_shape = tuple(int(s) for s in host_fake.shape)
            xs_node, padded_dim_size = self._emit_scan_form_reshape(
                compute_graph,
                ph,
                td,
                host_shape,
                block_size,
            )
            xs_nodes.append(xs_node)
            tiled_load_padded[(tname, td)] = padded_dim_size

        add_nodes: list[Node] = []
        for ai in additional_indices:
            outer_arg = outer_args[ai]
            assert isinstance(outer_arg, Node)
            if outer_arg not in node_map:
                raise exc.AutodiffNotSupported(
                    f"additional outer_arg {outer_arg.name} not in node_map"
                )
            add_nodes.append(node_map[outer_arg])
        for ln, tname in bcast_loads:
            if tname in tensor_to_placeholder:
                add_nodes.append(tensor_to_placeholder[tname])
            else:
                host_node = ln.args[0]
                host_fake = host_node.meta["val"]
                ph_name = f"tile_{tname}"
                ph = compute_graph.placeholder(ph_name)
                tensor_to_placeholder[tname] = ph
                input_mappings.append(
                    InputMapping(
                        placeholder_name=ph.name,
                        tensor_name=tname,
                        fake_tensor=host_fake,
                    )
                )
                add_nodes.append(ph)

        # When the scan dim was zero-padded (non-block-divisible), append a
        # per-step validity mask as an xs so the combine_fn can re-apply the
        # forward's OOB masking (_mask_to). Without it, padded positions
        # pollute reductions (e.g. attention's softmax) → wrong gradient.
        scan_mask_present = False
        if tiled_loads:
            ln0, tname0, td0 = tiled_loads[0]
            host_fake0 = ln0.args[0].meta["val"]
            dim0 = int(host_fake0.shape[td0])
            padded0 = tiled_load_padded.get((tname0, td0), dim0)
            if padded0 != dim0:
                n_blk = padded0 // block_size
                ar = compute_graph.call_function(
                    torch.ops.aten.arange.default,
                    (padded0,),
                    {"dtype": torch.int64, "device": host_fake0.device},
                )
                lt = compute_graph.call_function(torch.ops.aten.lt.Scalar, (ar, dim0))
                mask2d = compute_graph.call_function(
                    torch.ops.aten.view.default, (lt, [n_blk, block_size])
                )
                xs_nodes.append(mask2d)
                scan_mask_present = True

        combine_gm = self._build_combine_fn(
            sub_info,
            outer_args,
            carry_indices,
            additional_indices,
            tiled_loads,
            bcast_loads,
            stores,
            scan_mask_present,
        )

        # scan_op requires non-empty init; synthesize a unit carry.
        if not init_nodes:
            anchor = xs_nodes[0] if xs_nodes else (add_nodes[0] if add_nodes else None)
            assert anchor is not None
            anchor_fake = anchor.meta.get("val")
            device = (
                anchor_fake.device
                if isinstance(anchor_fake, torch.Tensor)
                else torch.device("cuda")
            )
            unit = compute_graph.call_function(
                torch.ops.aten.zeros.default,
                ([],),
                {"dtype": torch.float32, "device": device},
            )
            init_nodes = [unit]
            combine_gm = self._wrap_combine_fn_for_unit_carry(combine_gm)

        cg_idx = len(self.scan_combines)
        self.scan_combines.append(combine_gm)
        attr_name = f"scan_combine_graph_{cg_idx}"
        combine_attr = compute_graph.get_attr(attr_name)
        scan_node = compute_graph.call_function(
            _scan_op,
            (
                combine_attr,
                tuple(init_nodes),
                tuple(xs_nodes),
                tuple(add_nodes),
            ),
        )

        for i, ci in enumerate(carry_indices):
            oa = outer_args[ci]
            assert isinstance(oa, Node)
            phi = next(
                (
                    u
                    for u in oa.users
                    if getattr(u.target, "__name__", "") == "_phi" and u.args[0] is oa
                ),
                None,
            )
            if phi is None:
                continue
            getitem_node = compute_graph.call_function(operator.getitem, (scan_node, i))
            node_map[phi] = getitem_node

        n_carry = len(init_nodes)
        # Output layout: [*carry (n_carry), dummy_ys (1), *ys (len(stores))].
        # The dummy_ys at index n_carry ensures the backward scan always has
        # at least one grad_ys element (see _build_combine_fn comment).
        for j, (sn, tname, td, _val) in enumerate(stores):
            getitem_node = compute_graph.call_function(
                operator.getitem, (scan_node, n_carry + 1 + j)
            )
            host_node = sn.args[0]
            host_fake = host_node.meta["val"]
            host_shape = tuple(int(s) for s in host_fake.shape)
            # Look up any padding applied when the matching tiled load was
            # reshaped; pass it so the inverse can narrow back to original size.
            store_padded_dim_size = tiled_load_padded.get((tname, td))
            unreshaped = self._emit_scan_form_unreshape(
                compute_graph, getitem_node, td, host_shape, store_padded_dim_size
            )
            # The output finalization in `extract_computation_graph` reads
            # `meta["val"]` and `node_map[v]` on stored outputs.
            unreshaped.meta["val"] = host_fake
            node_map[unreshaped] = unreshaped
            tensor_current_value[tname] = unreshaped

    @staticmethod
    def _subscript_tile_dim(
        subscript: object,
        bs_name: str,
        bs_equiv_nodes: set[Node] | None = None,
    ) -> int | None:
        if not isinstance(subscript, (list, tuple)):
            return None
        for d, idx in enumerate(subscript):
            if not isinstance(idx, Node):
                continue
            if (
                getattr(idx.target, "__name__", "") == "_get_symnode"
                and idx.args[0] == bs_name
            ):
                return d
            if bs_equiv_nodes is not None and idx in bs_equiv_nodes:
                return d
        return None

    @staticmethod
    def _subscript_has_tile_id(subscript: object) -> bool:
        """Return True if any element of ``subscript`` is a ``tile_id`` call.

        This detects "point stores" that use ``t_i.id`` (an integer chunk
        index) as the scan-position dimension rather than the tiled subscript
        form.  Such stores are valid scan outputs but are indexed by a
        monotonically increasing integer derived from the scan iteration
        variable, not by a block-size symnode.
        """
        if not isinstance(subscript, (list, tuple)):
            return False
        for idx in subscript:
            if (
                isinstance(idx, Node)
                and getattr(idx.target, "__name__", "") == "tile_id"
            ):
                return True
        return False

    @staticmethod
    def _emit_scan_form_reshape(
        compute_graph: torch.fx.Graph,
        host_node: Node,
        tile_dim: int,
        fake_shape: tuple,
        block_size: int,
    ) -> tuple[Node, int]:
        """Reshape host_tensor so its tile_dim is split into
        (n_blocks, bk) and the n_blocks axis becomes leading.
        Pattern: ``view([…, n_blocks, bk, …]) → permute([tile_dim, …]) → clone(contig)``.

        For non-divisible tile dims, the tensor is zero-padded to the next
        multiple of block_size before reshaping.  The returned int is the
        padded dimension size (equal to the original when already divisible).
        """
        rank = len(fake_shape)
        dim_size = fake_shape[tile_dim]
        if not isinstance(dim_size, int):
            raise exc.AutodiffNotSupported(
                f"scan rewrite needs static tile dim; got size={dim_size!r}"
            )
        n_blocks = (dim_size + block_size - 1) // block_size
        padded_dim_size = n_blocks * block_size
        if padded_dim_size != dim_size:
            # Pad with zeros along tile_dim so the total size is divisible.
            # constant_pad_nd pads in reverse dimension order: for each dim
            # from last to first, provide (left_pad, right_pad).
            pad_amount = padded_dim_size - dim_size
            pad = [0] * (2 * (rank - 1 - tile_dim)) + [0, pad_amount]
            host_node = compute_graph.call_function(
                torch.ops.aten.constant_pad_nd.default, (host_node, pad, 0.0)
            )
        new_shape = [
            *fake_shape[:tile_dim],
            n_blocks,
            block_size,
            *fake_shape[tile_dim + 1 :],
        ]
        view = compute_graph.call_function(
            torch.ops.aten.view.default, (host_node, new_shape)
        )
        if tile_dim == 0:
            return (
                compute_graph.call_function(
                    torch.ops.aten.clone.default,
                    (view,),
                    {"memory_format": torch.contiguous_format},
                ),
                padded_dim_size,
            )
        perm = [tile_dim] + [d for d in range(rank + 1) if d != tile_dim]
        permuted = compute_graph.call_function(
            torch.ops.aten.permute.default, (view, perm)
        )
        return (
            compute_graph.call_function(
                torch.ops.aten.clone.default,
                (permuted,),
                {"memory_format": torch.contiguous_format},
            ),
            padded_dim_size,
        )

    @staticmethod
    def _emit_scan_form_unreshape(
        compute_graph: torch.fx.Graph,
        blocks_node: Node,
        tile_dim: int,
        target_shape: tuple,
        padded_dim_size: int | None = None,
    ) -> Node:
        """Inverse of `_emit_scan_form_reshape`.

        If ``padded_dim_size`` is given and differs from ``target_shape[tile_dim]``,
        the tensor is first viewed with the padded size and then narrowed (sliced)
        back to the original dimension size.
        """
        rank = len(target_shape)
        orig_dim_size = target_shape[tile_dim]
        needs_trim = padded_dim_size is not None and padded_dim_size != orig_dim_size
        # Build the intermediate shape that includes padding (if any).
        padded_shape = list(target_shape)
        if needs_trim:
            padded_shape[tile_dim] = padded_dim_size
        if tile_dim == 0:
            result = compute_graph.call_function(
                torch.ops.aten.view.default, (blocks_node, padded_shape)
            )
        else:
            inv_perm = [0] * (rank + 1)
            perm = [tile_dim] + [d for d in range(rank + 1) if d != tile_dim]
            for i, p in enumerate(perm):
                inv_perm[p] = i
            permed = compute_graph.call_function(
                torch.ops.aten.permute.default, (blocks_node, inv_perm)
            )
            cloned = compute_graph.call_function(
                torch.ops.aten.clone.default,
                (permed,),
                {"memory_format": torch.contiguous_format},
            )
            result = compute_graph.call_function(
                torch.ops.aten.view.default, (cloned, padded_shape)
            )
        if needs_trim:
            result = compute_graph.call_function(
                torch.ops.aten.narrow.default,
                (result, tile_dim, 0, orig_dim_size),
            )
        return result

    def _lower_combine_node(
        self,
        n: Node,
        cg: torch.fx.Graph,
        sub_node_map: dict[Node, Node],
        scalar_literal_map: dict[Node, object],
        store_nodes: set[Node],
        node_dtype_map: dict[Node, torch.dtype],
        matmul_ops: frozenset,
        get_node_dtype: Callable[[Node], torch.dtype | None],
        maybe_cast: Callable[[Node, torch.dtype], Node],
        mask_ph: Node | None,
    ) -> None:
        """Lower one Helion sub-graph node into the scan combine_fn graph ``cg``.

        The single shared per-node lowering used by both ``_build_combine_fn``
        (the main loop) and ``_inline_branch_nodes`` (per branch node): Helion
        tracing ops are aliased/resolved, ``hl.full``→``aten.full``,
        ``hl.dot``→a rank-appropriate matmul, ``subscript``→unsqueeze,
        ``_mask_to`` re-applies the OOB mask when a scan mask is present,
        ``eq``/``_if`` are statically resolved, and everything else is copied
        with matmul-dtype reconciliation.
        """
        if n.op in ("placeholder", "output"):
            return
        tn = getattr(n.target, "__name__", "")
        if tn == "_host_tensor" or tn == "load" or n in store_nodes:
            return
        if tn == "_mask_to" and mask_ph is not None:
            # Re-apply the forward's OOB masking inside the scan so padded
            # (non-block-divisible) positions don't pollute reductions like the
            # softmax max/sum. mask_ph is [block_size]; it broadcasts against
            # the value's trailing (key) dim.
            src = n.args[0]
            fill = n.args[1]
            if isinstance(src, Node) and src in sub_node_map:
                w = cg.call_function(
                    torch.ops.aten.where.ScalarOther,
                    (mask_ph, sub_node_map[src], fill),
                )
                if n.meta:
                    w.meta = n.meta.copy()
                    fake = n.meta.get("val")
                    if isinstance(fake, torch.Tensor):
                        node_dtype_map[w] = fake.dtype
                sub_node_map[n] = w
            return
        if tn in ("_new_var", "_mask_to"):
            src = n.args[0]
            if isinstance(src, Node) and src in sub_node_map:
                sub_node_map[n] = sub_node_map[src]
            return
        if tn == "_inductor_lowering_extra":
            data_inputs = n.args[0]
            src = (
                data_inputs[0]
                if isinstance(data_inputs, (list, tuple))
                else data_inputs
            )
            if isinstance(src, Node) and src in sub_node_map:
                sub_node_map[n] = sub_node_map[src]
            return
        if tn == "_phi":
            if isinstance(n.args[1], Node) and n.args[1] in sub_node_map:
                sub_node_map[n] = sub_node_map[n.args[1]]
            return
        if tn == "_get_symnode":
            sym_name = n.args[0]
            assert isinstance(sym_name, str)
            scalar_val = _resolve_symnode_expr(sym_name, self.scalar_values)
            if scalar_val is None and sym_name.startswith("block_size_"):
                try:
                    bid = int(sym_name[len("block_size_") :])
                    scalar_val = self.config_block_sizes.get(bid)
                except ValueError:
                    pass
            if scalar_val is not None:
                scalar_literal_map[n] = scalar_val
            return
        # Lower hl.full/hl.zeros to aten.full for AOT retrace.
        if (getattr(n.target, "__module__", "") or "").startswith(
            "helion.language"
        ) and tn == "full":
            shape_arg = n.args[0]
            fill_value = n.args[1]
            dtype_arg = n.args[2]
            assert isinstance(shape_arg, (list, tuple))
            assert isinstance(fill_value, (int, float))
            assert isinstance(dtype_arg, torch.dtype)
            resolved: list = []
            for s in shape_arg:
                if isinstance(s, Node) and s in sub_node_map:
                    resolved.append(sub_node_map[s])
                elif (
                    isinstance(s, Node)
                    and s.target is torch.ops.aten.sym_size.int
                    and isinstance(s.args[0], Node)
                    and s.args[0] in sub_node_map
                ):
                    ss = cg.call_function(
                        torch.ops.aten.sym_size.int,
                        (sub_node_map[s.args[0]], s.args[1]),
                    )
                    sub_node_map[s] = ss
                    resolved.append(ss)
                elif isinstance(s, int):
                    resolved.append(s)
                else:
                    resolved.append(s)
            fake = n.meta.get("val")
            dev = (
                fake.device if isinstance(fake, torch.Tensor) else torch.device("cuda")
            )
            fn = cg.call_function(
                torch.ops.aten.full.default,
                (resolved, fill_value),
                {"dtype": dtype_arg, "device": dev},
            )
            if n.meta:
                fn.meta = n.meta.copy()
            node_dtype_map[fn] = dtype_arg
            sub_node_map[n] = fn
            return
        if tn == "subscript":
            self._lower_subscript(n, cg, sub_node_map)
            return
        # Lower hl.dot → a rank-appropriate matmul (Helion matmul needs a
        # compile env). Promote operands to a shared dtype (aten matmul rejects
        # mixed-dtype inputs, unlike Triton).
        if tn == "dot":
            a0, a1 = n.args[0], n.args[1]
            assert isinstance(a0, Node) and isinstance(a1, Node)
            lhs = sub_node_map[a0]
            rhs = sub_node_map[a1]
            lhs_dt = get_node_dtype(lhs)
            rhs_dt = get_node_dtype(rhs)
            if lhs_dt is not None and rhs_dt is not None and lhs_dt != rhs_dt:
                target_dt = torch.promote_types(lhs_dt, rhs_dt)
                lhs = maybe_cast(lhs, target_dt)
                rhs = maybe_cast(rhs, target_dt)
            mm_op = self._matmul_op_for_ranks(a0, a1)
            new_node = cg.call_function(mm_op, (lhs, rhs))  # pyrefly: ignore [bad-argument-type]
            if n.meta:
                new_node.meta = n.meta.copy()
                fake = n.meta.get("val")
                if isinstance(fake, torch.Tensor):
                    node_dtype_map[new_node] = fake.dtype
            sub_node_map[n] = new_node
            return
        # Statically resolve comparisons (e.g. `eq(beta, 0.0)`).
        if tn == "eq" and len(n.args) == 2:
            a, b = n.args
            a_val = scalar_literal_map.get(a) if isinstance(a, Node) else a
            b_val = scalar_literal_map.get(b) if isinstance(b, Node) else b
            if a_val is not None and b_val is not None:
                scalar_literal_map[n] = bool(a_val == b_val)
                return
        # Statically resolve `_if`: inline the taken branch's nodes so no
        # Helion tracing ops leak through.
        if tn == "_if":
            cond_node = n.args[0]
            cond_val = (
                scalar_literal_map.get(cond_node)
                if isinstance(cond_node, Node)
                else cond_node
            )
            if cond_val is not None and isinstance(cond_val, bool):
                then_graph_id = n.args[1]
                else_graph_id = n.args[2]
                assert isinstance(then_graph_id, int)
                assert isinstance(else_graph_id, int)
                if cond_val:
                    branch_id = then_graph_id
                    branch_args = n.args[3]
                    result_offset = 0
                else:
                    branch_id = else_graph_id
                    branch_args = n.args[4]
                    # _if concatenates then outputs before else outputs; getitem
                    # indices for the else branch are offset by the then count.
                    then_info = self.all_graphs[then_graph_id]
                    then_out = next(
                        bn for bn in then_info.graph.nodes if bn.op == "output"
                    )
                    then_rets = then_out.args[0]
                    if not isinstance(then_rets, (list, tuple)):
                        then_rets = (then_rets,) if then_rets is not None else ()
                    result_offset = len(then_rets)
                assert isinstance(branch_args, (list, tuple))
                branch_info = self.all_graphs[branch_id]
                branch_phs = list(branch_info.graph.find_nodes(op="placeholder"))
                for bp, ba in zip(branch_phs, branch_args, strict=True):
                    if isinstance(ba, Node) and ba in sub_node_map:
                        sub_node_map[bp] = sub_node_map[ba]
                self._inline_branch_nodes(
                    branch_info,
                    cg,
                    sub_node_map,
                    scalar_literal_map,
                    store_nodes,
                    node_dtype_map,
                    matmul_ops,
                    get_node_dtype,
                    maybe_cast,
                    mask_ph,
                )
                branch_out = next(
                    bn for bn in branch_info.graph.nodes if bn.op == "output"
                )
                branch_rets = branch_out.args[0]
                if not isinstance(branch_rets, (list, tuple)):
                    branch_rets = (branch_rets,) if branch_rets is not None else ()
                for user in list(n.users):
                    if getattr(user.target, "__name__", "") == "getitem" and isinstance(
                        user.args[1], int
                    ):
                        idx = user.args[1] - result_offset
                        if 0 <= idx < len(branch_rets):
                            ret_node = branch_rets[idx]
                            if isinstance(ret_node, Node) and ret_node in sub_node_map:
                                sub_node_map[user] = sub_node_map[ret_node]
                return
        # Skip getitem on _if nodes (resolved above or dead branch).
        if tn == "getitem":
            container = n.args[0]
            if (
                isinstance(container, Node)
                and getattr(container.target, "__name__", "") == "_if"
            ):
                return
        # Generic op: substitute scalar literals, map nodes, reconcile matmul
        # operand dtypes, and copy.
        args = tuple(
            scalar_literal_map[a]
            if isinstance(a, Node) and a in scalar_literal_map
            else a
            for a in n.args
        )
        new_args = map_arg(args, sub_node_map.get)  # pyrefly: ignore [bad-specialization]
        new_kwargs = map_arg(dict(n.kwargs), sub_node_map.get)
        if n.target in matmul_ops:
            new_args_list = list(new_args)
            tensor_dtypes_nonnull = [
                get_node_dtype(a)
                for a in new_args_list
                if isinstance(a, Node) and get_node_dtype(a) is not None
            ]
            if tensor_dtypes_nonnull:
                target_dt = tensor_dtypes_nonnull[0]
                assert target_dt is not None
                for d in tensor_dtypes_nonnull[1:]:
                    assert d is not None
                    target_dt = torch.promote_types(target_dt, d)
                new_args_list = [
                    maybe_cast(a, target_dt) if isinstance(a, Node) else a
                    for a in new_args_list
                ]
            new_args = tuple(new_args_list)
        new_node = cg.call_function(n.target, new_args, new_kwargs)  # pyrefly: ignore [bad-argument-type]
        if n.meta:
            new_node.meta = n.meta.copy()
            fake = n.meta.get("val")
            if isinstance(fake, torch.Tensor):
                node_dtype_map[new_node] = fake.dtype
        sub_node_map[n] = new_node

    @staticmethod
    def _matmul_op_for_ranks(lhs_node: object, rhs_node: object) -> object:
        """Pick the aten matmul overload for ``hl.dot`` by operand rank:
        2-D→``mm``, 3-D→``bmm``, else ``matmul`` (handles broadcasting).
        ``hl.dot`` was previously hard-wired to ``bmm`` (3-D only), which
        rejected 2-D matmul accumulation loops.
        """

        def rank(x: object) -> int | None:
            f = x.meta.get("val") if isinstance(x, Node) else None
            return f.ndim if isinstance(f, torch.Tensor) else None

        lr, rr = rank(lhs_node), rank(rhs_node)
        if lr == 2 and rr == 2:
            return torch.ops.aten.mm.default
        if lr == 3 and rr == 3:
            return torch.ops.aten.bmm.default
        return torch.ops.aten.matmul.default

    def _build_combine_fn(
        self,
        sub_info: GraphInfo,
        outer_args: list,
        carry_indices: list[int],
        additional_indices: list[int],
        tiled_loads: list,
        bcast_loads: list,
        stores: list,
        scan_mask_present: bool = False,
    ) -> torch.fx.GraphModule:
        """Build a fresh FX `GraphModule` that's the combine_fn body for
        scan_op. Placeholders ordered ``[*carry, *xs_slice, *additional]``,
        outputs ``[*new_carry, *ys]``. Identity carries get an
        `aten.clone` to dodge scan_op's aliasing-check rejection.
        """
        cg = torch.fx.Graph()
        sub_phs = list(sub_info.graph.find_nodes(op="placeholder"))
        sub_node_map: dict[Node, Node] = {}
        # Track the dtype of each new-graph node so we can insert casts
        # before matmul-family ops that require uniform dtypes across all
        # operands (unlike Triton, standard PyTorch aten ops reject mixed
        # fp32/fp16/bf16 inputs to mm/bmm/addmm/baddbmm).
        node_dtype_map: dict[Node, torch.dtype] = {}

        def _get_node_dtype(node: Node) -> torch.dtype | None:
            """Return the dtype for a cg node, or None if unknown."""
            if node in node_dtype_map:
                return node_dtype_map[node]
            fake = node.meta.get("val") if node.meta else None
            if isinstance(fake, torch.Tensor):
                return fake.dtype
            return None

        carry_phs_in_cg: list[Node] = []
        for i, ci in enumerate(carry_indices):
            ph = cg.placeholder(f"carry_{i}")
            sub_node_map[sub_phs[ci]] = ph
            carry_phs_in_cg.append(ph)
            # Carry dtype from the sub-graph placeholder's fake tensor.
            src_fake = sub_phs[ci].meta.get("val") if sub_phs[ci].meta else None
            if isinstance(src_fake, torch.Tensor):
                node_dtype_map[ph] = src_fake.dtype
        xs_placeholders: list[Node] = []
        for i in range(len(tiled_loads)):
            xs_placeholders.append(cg.placeholder(f"x_slice_{i}"))
        # Per-step validity mask xs (present only when the scan dim was padded);
        # ordered right after the tiled xs to match the scan's xs tuple.
        mask_ph: Node | None = None
        if scan_mask_present:
            mask_ph = cg.placeholder("scan_mask")
        for i, ai in enumerate(additional_indices):
            add_ph = cg.placeholder(f"add_outer_{i}")
            sub_node_map[sub_phs[ai]] = add_ph
            src_fake = sub_phs[ai].meta.get("val") if sub_phs[ai].meta else None
            if isinstance(src_fake, torch.Tensor):
                node_dtype_map[add_ph] = src_fake.dtype
        bcast_placeholders: list[Node] = []
        for i in range(len(bcast_loads)):
            bcast_placeholders.append(cg.placeholder(f"add_load_{i}"))

        for i, (ln, _, _) in enumerate(tiled_loads):
            sub_node_map[ln] = xs_placeholders[i]
            # xs dtype from the load node's fake tensor.
            load_fake = ln.meta.get("val") if ln.meta else None
            if isinstance(load_fake, torch.Tensor):
                node_dtype_map[xs_placeholders[i]] = load_fake.dtype
        for i, (ln, _) in enumerate(bcast_loads):
            sub_node_map[ln] = bcast_placeholders[i]
            load_fake = ln.meta.get("val") if ln.meta else None
            if isinstance(load_fake, torch.Tensor):
                node_dtype_map[bcast_placeholders[i]] = load_fake.dtype

        # Matmul-family ops that require all tensor operands to share a dtype
        # in standard PyTorch (unlike Triton which allows mixed precision).
        _matmul_ops = frozenset(
            {
                torch.ops.aten.mm.default,
                torch.ops.aten.bmm.default,
                torch.ops.aten.addmm.default,
                torch.ops.aten.matmul.default,
                torch.ops.aten.baddbmm.default,
            }
        )

        def _maybe_cast(node: Node, target_dtype: torch.dtype) -> Node:
            """Insert an aten._to_copy cast if node's dtype != target_dtype."""
            cur = _get_node_dtype(node)
            if cur is None or cur == target_dtype:
                return node
            cast = cg.call_function(
                torch.ops.aten._to_copy.default, (node,), {"dtype": target_dtype}
            )
            node_dtype_map[cast] = target_dtype
            return cast

        store_nodes = {s[0] for s in stores}
        scalar_literal_map: dict[Node, object] = {}

        for n in sub_info.graph.nodes:
            self._lower_combine_node(
                n,
                cg,
                sub_node_map,
                scalar_literal_map,
                store_nodes,
                node_dtype_map,
                _matmul_ops,
                _get_node_dtype,
                _maybe_cast,
                mask_ph,
            )

        out_node = next(n for n in sub_info.graph.nodes if n.op == "output")
        sub_returns = out_node.args[0]
        if not isinstance(sub_returns, (list, tuple)):
            sub_returns = (sub_returns,) if sub_returns is not None else ()
        new_carry: list[Node] = []
        for i, ci in enumerate(carry_indices):
            # Use the _phi node's getitem index to find the correct sub_returns
            # position.  sub_returns is ordered by write-order (the order
            # variables were last assigned inside the scan loop body), while
            # carry_indices is in declaration order.  The _phi node in the
            # outer graph encodes the mapping: _phi.args[0] is the initial
            # outer_arg and _phi.args[1] is getitem(_for_loop, k) where k is
            # the write-order index into sub_returns.
            oa = outer_args[ci]
            sub_idx = i  # default: positional fallback
            if isinstance(oa, Node):
                phi = next(
                    (
                        u
                        for u in oa.users
                        if getattr(u.target, "__name__", "") == "_phi"
                        and u.args[0] is oa
                    ),
                    None,
                )
                if (
                    phi is not None
                    and isinstance(phi.args[1], Node)
                    and getattr(phi.args[1].target, "__name__", "") == "getitem"
                    and isinstance(phi.args[1].args[1], int)
                ):
                    sub_idx = phi.args[1].args[1]
            v = sub_node_map[sub_returns[sub_idx]]  # pyrefly: ignore [bad-index]
            # scan_op rejects combine_fns that return a placeholder
            # unchanged; clone identity carries to dodge the alias check.
            if v is carry_phs_in_cg[i]:
                v = cg.call_function(torch.ops.aten.clone.default, (v,))
            new_carry.append(v)
        ys = [sub_node_map[s[3]] for s in stores]
        # Add a dummy scalar ys (index n_carry + 0) so the backward scan always
        # has at least one grad_ys element.  scan_op backward builds
        # bw_xs = [grad_ys, saved_fw_xs, saved_intermediates]; if grad_ys is
        # empty (no stores, or xs not needed for recomputation), the backward
        # scan_op call does bw_xs[0].shape[0] to get scan_length and raises
        # IndexError.  The dummy ensures bw_xs is non-empty.
        dummy_device: object = torch.device("cuda")
        if carry_phs_in_cg:
            anchor_fake = carry_phs_in_cg[0].meta.get("val")
            if isinstance(anchor_fake, torch.Tensor):
                dummy_device = anchor_fake.device
        elif xs_placeholders:
            anchor_fake = xs_placeholders[0].meta.get("val")
            if isinstance(anchor_fake, torch.Tensor):
                dummy_device = anchor_fake.device
        dummy_ys = cg.call_function(
            torch.ops.aten.zeros.default,
            ([1],),
            {"dtype": torch.float32, "device": dummy_device},  # pyrefly: ignore [bad-argument-type]
        )
        cg.output(tuple([*new_carry, dummy_ys, *ys]))
        return torch.fx.GraphModule({}, cg)

    def _inline_branch_nodes(
        self,
        branch_info: GraphInfo,
        cg: torch.fx.Graph,
        sub_node_map: dict[Node, Node],
        scalar_literal_map: dict[Node, object],
        store_nodes: set[Node],
        node_dtype_map: dict[Node, torch.dtype],
        matmul_ops: frozenset,
        get_node_dtype: Callable[[Node], torch.dtype | None],
        maybe_cast: Callable[[Node, torch.dtype], Node],
        mask_ph: Node | None,
    ) -> None:
        """Inline a statically-taken ``_if`` branch's nodes into ``cg`` via the
        shared per-node lowering (recurses for nested ``_if``)."""
        for n in branch_info.graph.nodes:
            self._lower_combine_node(
                n,
                cg,
                sub_node_map,
                scalar_literal_map,
                store_nodes,
                node_dtype_map,
                matmul_ops,
                get_node_dtype,
                maybe_cast,
                mask_ph,
            )

    @staticmethod
    def _wrap_combine_fn_for_unit_carry(
        combine_gm: torch.fx.GraphModule,
    ) -> torch.fx.GraphModule:
        """Prepend a `unit` carry to combine_fn's signature; return it
        cloned (avoids scan_op aliasing). Used for no-carry loops.
        """
        old = combine_gm.graph
        new = torch.fx.Graph()
        node_map: dict[Node, Node] = {}
        unit_ph = new.placeholder("unit")
        for n in old.nodes:
            if n.op == "placeholder":
                node_map[n] = new.placeholder(n.name)
            elif n.op == "output":
                old_returns = n.args[0]
                if not isinstance(old_returns, (list, tuple)):
                    old_returns = (old_returns,)
                cloned = new.call_function(torch.ops.aten.clone.default, (unit_ph,))
                new_returns = (cloned, *map_arg(old_returns, node_map.get))
                new.output(new_returns)
            else:
                args = map_arg(n.args, node_map.get)
                kwargs = map_arg(n.kwargs, node_map.get)
                new_node = new.call_function(n.target, args, kwargs)  # pyrefly: ignore [bad-argument-type]
                if n.meta:
                    new_node.meta = n.meta.copy()
                node_map[n] = new_node
        return torch.fx.GraphModule({}, new)


def _strip_unsupported_kwargs(op: str, kwargs: dict) -> dict:
    """Drop kwargs Helion can't accept on factory/copy ops
    (``memory_format`` literals, ``layout``/``pin_memory``)."""
    out = dict(kwargs)
    if op in ("clone", "contiguous", "empty_like", "zeros_like", "ones_like"):
        out.pop("memory_format", None)
    if op in ("full", "empty", "zeros", "ones", "full_like", "arange", "scalar_tensor"):
        out.pop("layout", None)
        out.pop("pin_memory", None)
    return out


def _render_arg(arg: object, var_map: dict) -> str:
    """Render an FX arg to a Python source fragment, resolving Nodes via
    ``var_map``. The single generic arg renderer shared by every scan-path
    code emitter (replaces the former ``render``/``sub_render`` duplicates)."""
    if isinstance(arg, torch.fx.Node):
        return var_map[arg]
    if isinstance(arg, list):
        return "[" + ", ".join(_render_arg(a, var_map) for a in arg) + "]"
    if isinstance(arg, tuple):
        inner = ", ".join(_render_arg(a, var_map) for a in arg)
        return "(" + inner + ("," if len(arg) == 1 else "") + ")"
    if isinstance(arg, slice):
        return f"slice({arg.start!r}, {arg.stop!r}, {arg.step!r})"
    if isinstance(arg, float):
        if math.isinf(arg):
            return "float('-inf')" if arg < 0 else "float('inf')"
        if math.isnan(arg):
            return "float('nan')"
    if isinstance(arg, (torch.dtype, torch.layout, torch.memory_format)):
        return f"torch.{str(arg).split('.')[-1]}"
    if isinstance(arg, torch.device):
        return f"torch.device({str(arg)!r})"
    return repr(arg)


def _render_op(node: torch.fx.Node, var_map: dict) -> str:
    """Render a backward FX call_function node to a Helion-source RHS
    expression, resolving operands via ``var_map``. The single generic op
    renderer used by both the top-level backward walker and the combine_fn
    inliner (replaces the former ``render_op`` / ``_walk_combine_fn_body``
    duplicates).
    """
    target = node.target
    op_name = getattr(target, "_opname", None) or getattr(target, "__name__", "")
    if op_name == "iota":
        # prims.iota(length, *, start, step, dtype, device, ...) → torch.arange.
        # torch has no `iota`, so the generic dispatch would emit an invalid
        # `<length>.iota(...)`.
        length = node.args[0]
        kw = dict(node.kwargs)
        start = kw.pop("start", 0)
        step = kw.pop("step", 1)
        kw.pop("requires_grad", None)
        kw.pop("layout", None)
        kw.pop("pin_memory", None)
        kw_str = ", ".join(f"{k}={_render_arg(v, var_map)}" for k, v in kw.items())
        if isinstance(length, int) and isinstance(start, int) and isinstance(step, int):
            end = start + length * step
            base = f"torch.arange({start}, {end}, {step})"
        else:
            base = f"torch.arange({_render_arg(length, var_map)})"
        return base[:-1] + (", " + kw_str + ")" if kw_str else ")")
    if target is torch.ops.prims.rev.default:
        t_arg, dims = node.args
        t_meta = t_arg.meta.get("val") if isinstance(t_arg, torch.fx.Node) else None
        # `flip` on a size-1 dim is a no-op; alias instead.
        if (
            isinstance(t_meta, torch.Tensor)
            and isinstance(dims, list)
            and all(isinstance(d, int) and int(t_meta.shape[d]) == 1 for d in dims)
        ):
            return _render_arg(t_arg, var_map)
        return (
            f"torch.flip({_render_arg(t_arg, var_map)}, {_render_arg(dims, var_map)})"
        )
    args_r = [_render_arg(a, var_map) for a in node.args]
    if op_name == "alias":
        return args_r[0]
    if op_name == "copy" and len(args_r) >= 2:
        return f"{args_r[1]}.clone()"
    # Helion can't take a `torch.contiguous_format` literal; use `.contiguous()`.
    if (
        op_name == "clone"
        and node.kwargs.get("memory_format") is torch.contiguous_format
    ):
        return f"{args_r[0]}.contiguous()"
    kwargs = _strip_unsupported_kwargs(op_name, dict(node.kwargs))
    kwargs_r = [f"{k}={_render_arg(v, var_map)}" for k, v in kwargs.items()]
    # _to_copy / convert_element_type → tensor.to(dtype); dtype may be in args
    # (AOT bw graphs) or kwargs. With no dtype, alias the input.
    if op_name in ("_to_copy", "convert_element_type"):
        dtype_kw = node.kwargs.get("dtype")
        if dtype_kw is None and len(node.args) >= 2:
            dtype_kw = node.args[1]
        if isinstance(dtype_kw, torch.dtype):
            return f"{args_r[0]}.to(torch.{str(dtype_kw).split('.')[-1]})"
        return args_r[0]
    if op_name == "expand" and len(node.args) >= 2:
        # Concrete expand sizes fail against a symbolic tile dim; use -1 to keep
        # the existing size on non-broadcast (input dim != 1) dimensions.
        in_fake = (
            node.args[0].meta.get("val")
            if isinstance(node.args[0], torch.fx.Node)
            else None
        )
        size_arg = node.args[1]
        if isinstance(in_fake, torch.Tensor) and isinstance(size_arg, (list, tuple)):
            in_shape = list(in_fake.shape)
            adjusted = [
                "-1"
                if (isinstance(d, int) and i < len(in_shape) and in_shape[i] != 1)
                else _render_arg(d, var_map)
                for i, d in enumerate(size_arg)
            ]
            return f"{args_r[0]}.expand([{', '.join(adjusted)}])"
        return f"{args_r[0]}.expand({args_r[1]})"
    if op_name in ("view", "reshape") and len(node.args) >= 2:
        # Replace at most ONE non-1 concrete dim with -1 (view allows one -1)
        # so a symbolic tile dim can flow through.
        in_fake = (
            node.args[0].meta.get("val")
            if isinstance(node.args[0], torch.fx.Node)
            else None
        )
        size_arg = node.args[1]
        # Pure unsqueeze (target = input dims with size-1 dims inserted): emit
        # None-indexing instead of a literal view. A literal data dim (e.g.
        # view([-1, M, 1])) bakes an unbacked symint M that the tile-context
        # shape checker can't reconcile (non-pow-2 seqlens) — None-indexing
        # carries no literal sizes and sidesteps it entirely.
        if isinstance(in_fake, torch.Tensor) and isinstance(size_arg, (list, tuple)):
            non_unit = [d for d in size_arg if not (isinstance(d, int) and d == 1)]
            if len(non_unit) == in_fake.ndim and len(size_arg) > in_fake.ndim:
                idx = [
                    "None" if (isinstance(d, int) and d == 1) else ":" for d in size_arg
                ]
                return f"{args_r[0]}[{', '.join(idx)}]"
        if (
            isinstance(in_fake, torch.Tensor)
            and isinstance(size_arg, (list, tuple))
            and any(isinstance(d, int) and d > 1 for d in size_arg)
        ):
            in_shape = list(in_fake.shape)
            adjusted = [_render_arg(d, var_map) for d in size_arg]
            replaced = False
            for i, d in enumerate(size_arg):
                if replaced:
                    break
                if (
                    isinstance(d, int)
                    and d > 1
                    and (i >= len(in_shape) or in_shape[i] != 1)
                ):
                    adjusted[i] = "-1"
                    replaced = True
            return f"{args_r[0]}.{op_name}([{', '.join(adjusted)}])"
        return f"{args_r[0]}.{op_name}({args_r[1]})"
    if op_name in (
        "permute",
        "clone",
        "contiguous",
        "squeeze",
        "unsqueeze",
        "transpose",
        "flatten",
    ):
        return f"{args_r[0]}.{op_name}({', '.join(args_r[1:] + kwargs_r)})"
    if hasattr(torch, op_name):
        return f"torch.{op_name}({', '.join(args_r + kwargs_r)})"
    if args_r:
        return f"{args_r[0]}.{op_name}({', '.join(args_r[1:] + kwargs_r)})"
    return f"torch.ops.aten.{op_name}({', '.join(args_r + kwargs_r)})"


def _convert_scan_op_bw_to_helion(
    *,
    backward_graph: torch.fx.Graph,
    bw_module: torch.fx.GraphModule | None,
    input_mappings: list,
    input_tensors: tuple[torch.Tensor, ...],
    init_specs: list,
    grad_out_shapes: tuple[tuple[int, ...], ...],
    compute_graph: torch.fx.Graph | None = None,
) -> str:
    """Lower a bw FX (with `scan_op` nodes) to a single `@helion.kernel`
    backward source. Generated signature:
    ``backward_kernel(grad_out, *kernel_inputs, *_init_<i>, *grad_<input>)``;
    gradients flow back via the pre-allocated ``grad_<input>`` buffers.
    """

    assert bw_module is not None, "bw_module needed for scan_op submodule lookup"

    placeholders = [n for n in backward_graph.nodes if n.op == "placeholder"]
    out_node = next(n for n in backward_graph.nodes if n.op == "output")

    n_inits = len(init_specs)
    kernel_input_names = [m.tensor_name for m in input_mappings]

    init_name_by_ph: dict[str, str] = {}
    for i, spec in enumerate(init_specs):
        init_name_by_ph[spec.placeholder_name] = f"_init_{i}"
    input_name_by_ph: dict[str, str] = {}
    for m in input_mappings:
        input_name_by_ph[m.placeholder_name] = m.tensor_name

    primal_names: list[str] = []
    if compute_graph is not None:
        for ph in compute_graph.find_nodes(op="placeholder"):
            if ph.name in init_name_by_ph:
                primal_names.append(init_name_by_ph[ph.name])
            elif ph.name in input_name_by_ph:
                primal_names.append(input_name_by_ph[ph.name])
            else:
                raise exc.AutodiffNotSupported(
                    f"unmapped compute_graph placeholder {ph.name!r}"
                )
    else:
        primal_names = list(kernel_input_names) + [f"_init_{i}" for i in range(n_inits)]

    def _primal_var(idx: int) -> str:
        return primal_names[idx - 1]

    def _tangent_var(idx: int) -> str:
        if len(grad_out_shapes) == 1:
            return "grad_out"
        return f"grad_out_{idx - 1}"

    var_counter = {"v": 0}

    def fresh(hint: str) -> str:
        var_counter["v"] += 1
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in hint)
        if not safe or safe[0].isdigit():
            safe = "v_" + safe
        return f"{safe}_{var_counter['v']}"

    node_to_var: dict[torch.fx.Node, str] = {}

    for ph in placeholders:
        if ph.name.startswith("primals_"):
            idx = int(ph.name.split("_")[1])
            node_to_var[ph] = _primal_var(idx)
        elif ph.name.startswith("tangents_"):
            idx = int(ph.name.split("_")[1])
            node_to_var[ph] = _tangent_var(idx)
        else:
            node_to_var[ph] = ph.name

    def render(arg: object) -> str:
        return _render_arg(arg, node_to_var)

    def render_op(node: torch.fx.Node) -> str:
        return _render_op(node, node_to_var)

    grad_buffer_names: list[str] = [f"grad_{nm}" for nm in kernel_input_names]

    indent = "    "
    device_loop_count = [0]
    # host_lines: kernel-host code emitted before the device wrapper.
    # body: device-wrapper interior (one or more scan loops).
    # post_lines: kernel-host code emitted after the wrapper, operating
    # on filled buffers (post-scan reshape + grad_buf assignments).
    host_lines: list[str] = []
    body: list[str] = []
    post_lines: list[str] = []
    post_lines_inside: list[str] = []  # must go inside wrapper

    def _walk_combine_fn_body(
        combine_gm: torch.fx.GraphModule,
        carry_vars: list[str],
        xs_vars: list[str],
        add_vars: list[str],
        depth: str,
    ) -> tuple[list[str], list[str | None]]:
        """Inline combine_fn body; return (lines, output_vars) where the
        output vars correspond to the ``(*new_carry, *ys)`` return tuple.
        """
        sub_node_to_var: dict[torch.fx.Node, str] = {}
        sub_lines: list[str] = []
        sub_phs = list(combine_gm.graph.find_nodes(op="placeholder"))
        n_carry = len(carry_vars)
        n_xs = len(xs_vars)
        for i, ph in enumerate(sub_phs):
            if i < n_carry:
                sub_node_to_var[ph] = carry_vars[i]
            elif i < n_carry + n_xs:
                sub_node_to_var[ph] = xs_vars[i - n_carry]
            else:
                sub_node_to_var[ph] = add_vars[i - n_carry - n_xs]

        for n in combine_gm.graph.nodes:
            if n.op in ("placeholder", "output"):
                continue
            op_name = getattr(n.target, "_opname", None) or getattr(
                n.target, "__name__", ""
            )
            v = fresh(op_name or "v")
            sub_node_to_var[n] = v
            sub_lines.append(f"{depth}{v} = {_render_op(n, sub_node_to_var)}")

        sub_out_node = next(n for n in combine_gm.graph.nodes if n.op == "output")
        rets = sub_out_node.args[0]
        if not isinstance(rets, (list, tuple)):
            rets = (rets,)
        # None entries in rets represent "no gradient" for non-differentiable
        # outputs (from AOT backward combine_fn).  Use a sentinel value ``None``
        # so callers can detect and skip updating the corresponding carry or ys.
        out_vars = [sub_node_to_var[r] if r is not None else None for r in rets]  # pyrefly: ignore [bad-index]
        return sub_lines, out_vars

    out_args = out_node.args[0]
    if not isinstance(out_args, (list, tuple)):
        out_args = (out_args,)
    # Map: kernel_input name → grad buffer name + index in `tensor_inputs`.
    name_to_grad_buf: dict[str, tuple[str, int]] = {
        nm: (f"grad_{nm}", i) for i, nm in enumerate(kernel_input_names)
    }

    # Pre-pass: classify each node as "device-touching" (depends on a
    # scan_op output) vs "pure host" (doesn't). Pure-host ops emit
    # before the wrapper at the kernel-host level; device-touching ops
    # emit inside the wrapper.
    #
    # Reversed-buffer boundary: a ``prims.rev`` over dim 0 of a scan ys getitem
    # is materialized into a host buffer (``ys_rev_buf``) written inside the
    # loop and read afterwards. The read + downstream reshape is a HOST op that
    # must run AFTER the device loop, so the device taint stops at this boundary.
    rev_boundary_nodes: set[torch.fx.Node] = set()
    for n in backward_graph.nodes:
        if not (
            n.op == "call_function"
            and n.target is torch.ops.prims.rev.default
            and len(n.args) >= 2
            and n.args[1] == [0]
            and isinstance(n.args[0], torch.fx.Node)
            and n.args[0].op == "call_function"
            and n.args[0].target is operator.getitem
            and isinstance(n.args[0].args[0], torch.fx.Node)
            and n.args[0].args[0].target is _scan_op
        ):
            continue
        gi = n.args[0]
        assert isinstance(gi, Node)
        scan_node = gi.args[0]
        assert isinstance(scan_node, Node)
        idx = gi.args[1]
        n_carry = (
            len(scan_node.args[1])
            if isinstance(scan_node.args[1], (list, tuple))
            else 0
        )
        gi_meta = gi.meta.get("val")
        # Only ys getitems (idx >= n_carry) with a real (>1) scan dim get
        # rev-buffered; size-1 revs are no-ops handled inline elsewhere.
        if (
            isinstance(idx, int)
            and idx >= n_carry
            and isinstance(gi_meta, torch.Tensor)
            and gi_meta.dim() >= 1
            and int(gi_meta.shape[0]) > 1
        ):
            rev_boundary_nodes.add(n)

    # Carry-output getitems: the final value of a scan carry is a host buffer
    # (clone of the init, updated in the loop) that is materialized after the
    # loop. A grad output sourced from one must be written at host scope
    # (post_lines), not inside the wrapper where the host buffer isn't loaded.
    carry_output_nodes: set[torch.fx.Node] = set()
    for n in backward_graph.nodes:
        if n.op == "call_function" and n.target is _scan_op:
            n_carry = len(n.args[1]) if isinstance(n.args[1], (list, tuple)) else 0
            for u in n.users:
                if (
                    u.op == "call_function"
                    and u.target is operator.getitem
                    and isinstance(u.args[1], int)
                    and u.args[1] < n_carry
                ):
                    carry_output_nodes.add(u)

    device_nodes: set[torch.fx.Node] = set()
    for n in backward_graph.nodes:
        if n.op == "call_function" and n.target is _scan_op:
            device_nodes.add(n)
            for u in n.users:
                if u.op == "call_function" and u.target is operator.getitem:
                    device_nodes.add(u)
    # Propagate forward: any op that consumes a device node becomes one.
    changed = True
    while changed:
        changed = False
        for n in backward_graph.nodes:
            if n in device_nodes:
                continue
            if n.op != "call_function":
                continue
            for arg in n.all_input_nodes:
                if arg in device_nodes:
                    device_nodes.add(n)
                    changed = True
                    break

    # Inter-scan nodes: device-touched nodes that transitively feed into
    # a later scan_op. These must stay inside the device wrapper (body),
    # not in post_lines (which emit after the wrapper).
    inter_scan_nodes: set[torch.fx.Node] = set()
    for n in backward_graph.nodes:
        if n.op == "call_function" and n.target is _scan_op:
            for arg in n.all_input_nodes:
                if arg in device_nodes:
                    inter_scan_nodes.add(arg)
    changed = True
    while changed:
        changed = False
        for n in backward_graph.nodes:
            if n not in device_nodes or n in inter_scan_nodes:
                continue
            for u in n.users:
                if u in inter_scan_nodes:
                    inter_scan_nodes.add(n)
                    changed = True
                    break

    # Post-host nodes: the reshape chain downstream of a reversed-buffer
    # boundary. They read a fully materialized host buffer and must emit AFTER
    # the device loop (post_lines), not inside it. A node qualifies only if
    # every device-tainted input reaches it through the reversed-buffer chain
    # (no live wrapper dependency) and it does not feed a later scan.
    post_host_nodes: set[torch.fx.Node] = set()
    changed = True
    while changed:
        changed = False
        for n in backward_graph.nodes:
            if (
                n.op != "call_function"
                or n in post_host_nodes
                or n in rev_boundary_nodes
                or n in inter_scan_nodes
            ):
                continue
            ins = n.all_input_nodes
            if not any(a in rev_boundary_nodes or a in post_host_nodes for a in ins):
                continue
            if all(
                a not in device_nodes or a in rev_boundary_nodes or a in post_host_nodes
                for a in ins
            ):
                post_host_nodes.add(n)
                changed = True

    def _device_str(d: object) -> str:
        if isinstance(d, torch.device):
            return f"torch.device({str(d)!r})"
        return repr(d)

    def _dtype_str(d: torch.dtype) -> str:
        return f"torch.{str(d).split('.')[-1]}"

    def _find_getitem(scan_node: torch.fx.Node, idx: int) -> torch.fx.Node | None:
        for u in scan_node.users:
            if (
                u.op == "call_function"
                and u.target is operator.getitem
                and u.args[1] == idx
            ):
                return u
        return None

    for n in backward_graph.nodes:
        if n.op != "call_function":
            continue
        target = n.target

        if target is _scan_op:
            combine_attr_node, init_tuple, xs_tuple, add_tuple = n.args
            assert combine_attr_node.op == "get_attr"  # pyrefly: ignore [missing-attribute]
            attr_name = combine_attr_node.target  # pyrefly: ignore [missing-attribute]
            combine_gm = getattr(bw_module, attr_name)

            xs_origins: list[tuple] = []
            for x in xs_tuple:  # pyrefly: ignore [not-iterable]
                x_meta = x.meta.get("val")  # pyrefly: ignore [missing-attribute]
                if not isinstance(x_meta, torch.Tensor):
                    raise exc.AutodiffNotSupported(
                        f"xs of {attr_name!r} has no fake-tensor meta"
                    )
                if x_meta.dim() < 1:
                    raise exc.AutodiffNotSupported(
                        f"xs of {attr_name!r} must be at least 1D"
                    )
                xs_origins.append((x, int(x_meta.shape[0]), list(x_meta.shape[1:])))

            scan_n_blocks = xs_origins[0][1]
            for _x_node, nb, _ in xs_origins[1:]:
                if nb != scan_n_blocks:
                    raise exc.AutodiffNotSupported(
                        f"mismatched n_blocks among xs of {attr_name!r}"
                    )

            block_size = 1
            scan_dim_size = scan_n_blocks
            tile_var = fresh("tile")
            n_carry = len(init_tuple)  # pyrefly: ignore [bad-argument-type]
            # n_blocks=1 reduces the scan to a single iteration; inline the
            # body to side-step tile.id-as-index unbacked-SymInt limitations.
            inline = scan_n_blocks == 1

            carry_vars: list[str] = []
            carry_ranks: list[int] = []
            for init_node in init_tuple:  # pyrefly: ignore [not-iterable]
                init_fake = init_node.meta.get("val")  # pyrefly: ignore [missing-attribute]
                rank = init_fake.dim() if isinstance(init_fake, torch.Tensor) else 0
                carry_ranks.append(rank)
                if inline:
                    carry_vars.append(render(init_node))
                else:
                    v = fresh("carry")
                    carry_vars.append(v)
                    if init_node in device_nodes:
                        # Device-derived init: may reference a host tensor
                        # inside the wrapper — load via subscript first.
                        init_var = render(init_node)
                        init_meta = init_node.meta.get("val")  # pyrefly: ignore [missing-attribute]
                        if isinstance(init_meta, torch.Tensor) and init_meta.dim() > 0:
                            loaded = fresh("init_loaded")
                            idx = ", ".join([":"] * init_meta.dim())
                            body.extend(
                                [
                                    f"{indent}{loaded} = {init_var}[{idx}]",
                                    f"{indent}{v} = {loaded}.clone()",
                                ]
                            )
                        else:
                            body.append(f"{indent}{v} = {init_var}.clone()")
                    else:
                        host_lines.append(f"{indent}{v} = {render(init_node)}.clone()")

            if not inline:
                for i in range(n_carry):
                    gi = _find_getitem(n, i)
                    if gi is not None:
                        node_to_var[gi] = carry_vars[i]

            cf_out_node = next(nn for nn in combine_gm.graph.nodes if nn.op == "output")
            cf_rets = cf_out_node.args[0]
            if not isinstance(cf_rets, (list, tuple)):
                cf_rets = (cf_rets,)
            n_ys = len(cf_rets) - n_carry
            if n_ys < 0:
                raise exc.AutodiffNotSupported(
                    f"backward combine_fn of {attr_name!r} has {len(cf_rets)} "
                    f"outputs but n_carry={n_carry}; expected at least n_carry outputs"
                )

            ys_buf_vars: list[str | None] = []
            ys_buf_ranks: list[int] = []
            for j in range(n_ys):
                gi = _find_getitem(n, n_carry + j)
                if gi is None:
                    ys_buf_vars.append(None)
                    ys_buf_ranks.append(0)
                    continue
                gi_meta = gi.meta.get("val")
                if not isinstance(gi_meta, torch.Tensor):
                    ys_buf_vars.append(None)
                    ys_buf_ranks.append(0)
                    continue
                shape = list(gi_meta.shape)
                buf = fresh("ys_buf")
                host_lines.append(
                    f"{indent}{buf} = torch.empty({shape}, "
                    f"dtype={_dtype_str(gi_meta.dtype)}, "
                    f"device={_device_str(gi_meta.device)})"
                )
                ys_buf_vars.append(buf)
                ys_buf_ranks.append(len(shape))
                node_to_var[gi] = buf

            if inline:
                inner = indent
                scan_idx_expr = None
            else:
                body.append(
                    f"{indent}for {tile_var} in hl.tile({scan_dim_size}, "
                    f"block_size={block_size}):"
                )
                inner = indent + "    "
                scan_idx_expr = f"{tile_var}.id"
                device_loop_count[0] += 1

            xs_slice_vars: list[str] = []
            for x_node, _, inner_shape in xs_origins:
                origin_var = node_to_var[x_node]
                if scan_idx_expr is None:
                    full_slice_keep = ", ".join([":"] * (len(inner_shape) + 1))
                    if x_node in device_nodes:
                        loaded = fresh("xs_load")
                        body.append(
                            f"{inner}{loaded} = {origin_var}[{full_slice_keep}]"
                        )
                        v = fresh("xs_slice")
                        body.append(f"{inner}{v} = {loaded}.squeeze(0)")
                    else:
                        sq = fresh("xs_sq")
                        host_lines.append(f"{indent}{sq} = {origin_var}.squeeze(0)")
                        full_slice = ", ".join([":"] * len(inner_shape)) or ":"
                        v = fresh("xs_slice")
                        body.append(f"{inner}{v} = {sq}[{full_slice}]")
                    xs_slice_vars.append(v)
                else:
                    slice_indices = [scan_idx_expr]
                    slice_indices.extend([":"] * len(inner_shape))
                    slice_expr = f"{origin_var}[{', '.join(slice_indices)}]"
                    v = fresh("xs_slice")
                    body.append(f"{inner}{v} = {slice_expr}")
                    xs_slice_vars.append(v)

            # Helion forbids direct host-tensor use in device code: rank>0
            # carries / additional inputs must be loaded via subscript.
            cur_carry_vars: list[str] = []
            for i, (ci, rank) in enumerate(zip(carry_vars, carry_ranks, strict=True)):
                cv = fresh("cur_carry")
                if (inline and init_tuple[i] in device_nodes) or rank == 0:  # pyrefly: ignore [bad-index, unsupported-operation]
                    body.append(f"{inner}{cv} = {ci}")
                elif init_tuple[i] in device_nodes:  # pyrefly: ignore [bad-index, unsupported-operation]
                    # Device-derived carry: alias (already a device tensor).
                    body.append(f"{inner}{cv} = {ci}")
                else:
                    idx = ", ".join([":"] * rank)
                    body.append(f"{inner}{cv} = {ci}[{idx}]")
                cur_carry_vars.append(cv)

            add_load_vars: list[str] = []
            for add_node in add_tuple:  # pyrefly: ignore [not-iterable]
                a_fake = add_node.meta.get("val")  # pyrefly: ignore [missing-attribute]
                a_rank = a_fake.dim() if isinstance(a_fake, torch.Tensor) else 0
                if a_rank == 0:
                    add_load_vars.append(render(add_node))
                else:
                    av = fresh("add_load")
                    idx = ", ".join([":"] * a_rank)
                    body.append(f"{inner}{av} = {render(add_node)}[{idx}]")
                    add_load_vars.append(av)

            sub_lines, out_vars = _walk_combine_fn_body(
                combine_gm, cur_carry_vars, xs_slice_vars, add_load_vars, inner
            )
            body.extend(sub_lines)

            new_carry_vars = out_vars[:n_carry]
            ys_vars = out_vars[n_carry:]

            if inline:
                for i in range(n_carry):
                    gi = _find_getitem(n, i)
                    nv = new_carry_vars[i]
                    if gi is not None and nv is not None:
                        node_to_var[gi] = nv
            else:
                for ci, nv, rank, init_node in zip(
                    carry_vars,
                    new_carry_vars,
                    carry_ranks,
                    init_tuple,  # pyrefly: ignore [bad-argument-type]
                    strict=True,
                ):
                    if nv is None:
                        # None means "no gradient for this carry" — leave ci unchanged.
                        continue
                    if rank == 0:
                        # Rank-0 (scalar) carries must be updated via subscript
                        # assignment to avoid CannotModifyHostVariableOnDevice.
                        # Helion forbids rebinding a host variable to a device
                        # value inside a tile loop (ast.Name LHS check), but
                        # subscript assignment (ast.Subscript LHS) is allowed.
                        # For a 0-dim tensor, [()] is the canonical subscript.
                        body.append(f"{inner}{ci}[()] = {nv}")
                    elif init_node in device_nodes:
                        # Device-derived carry: rebind (subscript assign
                        # on device tensors is forbidden by Helion).
                        body.append(f"{inner}{ci} = {nv}")
                    else:
                        idx = ", ".join([":"] * rank)
                        body.append(f"{inner}{ci}[{idx}] = {nv}")

            for yi_var, buf, rank in zip(
                ys_vars, ys_buf_vars, ys_buf_ranks, strict=True
            ):
                if buf is None or yi_var is None:
                    # None yi_var: "no gradient for this ys" from backward AOT.
                    continue
                if scan_idx_expr is None:
                    if rank == 0:
                        # Scalar ys can't be full-sliced; write with [()] subscript.
                        body.append(f"{inner}{buf}[()] = {yi_var}")
                    else:
                        # Helion can't take a 0 literal subscript; full-slice
                        # the leading-1 buffer with an unsqueezed ys instead.
                        full_slice = ", ".join([":"] * rank)
                        body.append(
                            f"{inner}{buf}[{full_slice}] = {yi_var}.unsqueeze(0)"
                        )
                else:
                    if rank == 0:
                        # Scalar ys buffer can't be indexed by tile.id; skip.
                        pass
                    else:
                        rest = ", ".join([":"] * (rank - 1)) if rank > 1 else ""
                        idx = scan_idx_expr + ((", " + rest) if rest else "")
                        body.append(f"{inner}{buf}[{idx}] = {yi_var}")
                    if rank == 0:
                        # Rank-0 ys can't use rev-buffer pattern; skip.
                        gi = None
                    else:
                        # If a rev(getitem, [0]) consumer exists, pre-write ys
                        # in reversed order to a separate buffer and alias the
                        # rev node so tile.id indexing on a local is avoided.
                        gi = _find_getitem(n, n_carry + ys_vars.index(yi_var))
                    if gi is not None:
                        for rev_u in gi.users:
                            if (
                                rev_u.op == "call_function"
                                and rev_u.target is torch.ops.prims.rev.default
                                and rev_u.args[1] == [0]
                            ):
                                rev_buf = fresh("ys_rev_buf")
                                shape_str = f"[{', '.join(str(s) for s in gi.meta['val'].shape)}]"
                                host_lines.append(
                                    f"{indent}{rev_buf} = torch.empty("
                                    f"{shape_str}, dtype={_dtype_str(gi.meta['val'].dtype)}, "
                                    f"device={_device_str(gi.meta['val'].device)})"
                                )
                                rev_rest = (
                                    ", ".join([":"] * (rank - 1)) if rank > 1 else ""
                                )
                                rev_idx = f"{scan_n_blocks} - 1 - {scan_idx_expr}" + (
                                    (", " + rev_rest) if rev_rest else ""
                                )
                                body.append(f"{inner}{rev_buf}[{rev_idx}] = {yi_var}")
                                node_to_var[rev_u] = rev_buf
            continue

        if (
            target is operator.getitem
            and isinstance(n.args[0], torch.fx.Node)
            and n.args[0].target is _scan_op
        ):
            continue

        # Already pre-mapped (e.g., reversed-write ys_buf from scan handler).
        if n in node_to_var:
            continue

        # `prims.rev` on a size-1 dim is a no-op; alias to skip emitting it.
        if target is torch.ops.prims.rev.default:
            t_arg, dims = n.args
            t_meta = t_arg.meta.get("val") if isinstance(t_arg, torch.fx.Node) else None
            if (
                isinstance(t_meta, torch.Tensor)
                and isinstance(dims, list)
                and all(isinstance(d, int) and int(t_meta.shape[d]) == 1 for d in dims)
            ):
                node_to_var[n] = node_to_var[t_arg]  # pyrefly: ignore [bad-index]
                continue

        v = fresh(getattr(target, "__name__", "v"))
        node_to_var[n] = v
        saved_vars: dict[torch.fx.Node, str] = {}
        if n in post_host_nodes:
            # Host-scope reshape on a materialized reversed buffer: emit after
            # the device loop (post_lines), operating on the full tensor (no
            # subscript loading).
            target_list = post_lines
        elif n not in device_nodes:
            target_list = host_lines
        elif n in inter_scan_nodes:
            # Load host-tensor args before using in device wrapper.
            # Save originals so non-device nodes later still see host names.
            for arg_node in n.all_input_nodes:
                arg_meta = arg_node.meta.get("val")
                if not isinstance(arg_meta, torch.Tensor) or arg_meta.dim() == 0:
                    continue
                arg_var = node_to_var.get(arg_node)
                if arg_var is None or arg_var.startswith("loaded_"):
                    continue
                saved_vars[arg_node] = arg_var
                loaded = fresh("loaded")
                idx = ", ".join([":"] * arg_meta.dim())
                body.append(f"{indent}{loaded} = {arg_var}[{idx}]")
                node_to_var[arg_node] = loaded
            target_list = body
        else:
            # If any input is from device scope, the op needs wrapper.
            needs_wrapper = any(
                arg in inter_scan_nodes or arg in device_nodes
                for arg in n.all_input_nodes
            )
            if needs_wrapper:
                # Pre-load ALL tensor inputs (host tensors need subscript
                # loading inside the wrapper; device tensors get a no-op view).
                # Save originals so non-device nodes later still see host names.
                for arg_node in n.all_input_nodes:
                    arg_meta = arg_node.meta.get("val")
                    if not isinstance(arg_meta, torch.Tensor) or arg_meta.dim() == 0:
                        continue
                    arg_var = node_to_var.get(arg_node)
                    if arg_var is None or arg_var.startswith("loaded_"):
                        continue
                    saved_vars[arg_node] = arg_var
                    loaded = fresh("loaded")
                    idx = ", ".join([":"] * arg_meta.dim())
                    post_lines_inside.append(f"{indent}{loaded} = {arg_var}[{idx}]")
                    node_to_var[arg_node] = loaded
                target_list = post_lines_inside
            else:
                target_list = post_lines
        target_list.append(f"{indent}{v} = {render_op(n)}")
        # Restore original var names so subsequent host-scope nodes
        # don't reference wrapper-scoped loaded_* variables.
        if n in device_nodes and saved_vars:
            node_to_var.update(saved_vars)

    for pos, val in enumerate(out_args):
        if not isinstance(val, torch.fx.Node):
            continue
        if pos >= len(primal_names):
            continue
        pname = primal_names[pos]
        if pname not in name_to_grad_buf:
            continue
        buf_name, kin_idx = name_to_grad_buf[pname]
        rank = input_tensors[kin_idx].dim()
        # If source is a device-scoped node, the write must be inside wrapper.
        # Reversed-buffer (post-host) and carry-final sources are materialized
        # after the loop, so their grad store belongs at host scope (post_lines).
        val_in_wrapper = (
            val in device_nodes
            and val not in post_host_nodes
            and val not in carry_output_nodes
        )
        target = post_lines_inside if val_in_wrapper else post_lines
        if rank == 0:
            # Use [()] for scalar tensor write; [None] would unsqueeze instead.
            target.append(f"{indent}{buf_name}[()] = {render(val)}")
        else:
            idx = ", ".join([":"] * rank)
            target.append(f"{indent}{buf_name}[{idx}] = {render(val)}")

    sig_params = [
        "grad_out" if len(grad_out_shapes) == 1 else f"grad_out_{i}"
        for i in range(len(grad_out_shapes))
    ]
    sig_params += list(kernel_input_names)
    sig_params += [f"_init_{i}" for i in range(n_inits)]
    sig_params += grad_buffer_names

    src_lines = [
        '"""',
        "Auto-generated Helion backward kernel (scan-hop).",
        '"""',
        "",
        "import torch",
        "import helion",
        "from helion import language as hl",
        "",
        (
            "@helion.kernel(autotune_effort='none', static_shapes=True,"
            " ignore_warnings=[helion.exc.TensorOperationInWrapper])"
        ),
        f"def backward_kernel({', '.join(sig_params)}):",
    ]
    src_lines.extend(host_lines)
    # Helion requires exactly one top-level device loop; wrap the body in
    # a 1-iter outer loop when zero or more than one loop was emitted.
    if device_loop_count[0] == 0:
        # Inline path (all scans had n_blocks==1): body was emitted without
        # any tile loop.  Add a 1-iter wrapper to satisfy Helion's device-loop
        # requirement, but keep post_lines_inside at the host level so that
        # view/reshape ops with concrete shapes don't see symbolic tile dims.
        src_lines.append(f"{indent}for _outer in hl.tile(1, block_size=1):")
        for line in body:
            src_lines.append(indent + line)
        src_lines.extend(post_lines_inside)
        src_lines.extend(post_lines)
    elif device_loop_count[0] != 1:
        # Multiple scan loops: wrap everything together.
        src_lines.append(f"{indent}for _outer in hl.tile(1, block_size=1):")
        for line in body:
            src_lines.append(indent + line)
        for line in post_lines_inside:
            src_lines.append(indent + line)
        src_lines.extend(post_lines)
    else:
        src_lines.extend(body)
        src_lines.extend(post_lines_inside)
        src_lines.extend(post_lines)
    src_lines.append(f"{indent}return")
    return "\n".join(src_lines)


def _rewrite_convert_element_type(graph: torch.fx.Graph) -> None:
    """Replace ``prims.convert_element_type`` nodes with
    ``aten._to_copy`` in-place.  ``convert_element_type`` has no
    autograd formula and adding it to the decomposition table causes
    an infinite recursion (``_to_copy`` decomposes back to
    ``convert_element_type``).  Rewriting before AOT tracing avoids
    the cycle entirely.
    """
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.prims.convert_element_type.default
        ):
            # prims.convert_element_type(x, dtype) → aten._to_copy(x, dtype=dtype)
            x_arg = node.args[0]
            dtype_arg = node.args[1]
            with graph.inserting_after(node):
                new_node = graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (x_arg,),
                    {"dtype": dtype_arg},
                )
                new_node.meta = node.meta.copy() if node.meta else {}
                node.replace_all_uses_with(new_node)
            graph.erase_node(node)


def _rewrite_reverse_cumsum(graph: torch.fx.Graph) -> bool:
    """Rewrite cumsum's backward ``rev(cumsum(rev(g, [d]), d), [d])`` into the
    flip-free algebraic form ``g.sum(d, keepdim=True) - cumsum(g, d) + g``.

    AOT differentiates ``aten.cumsum`` using ``prims.rev`` (reverse-scan), but
    Helion's ``flip`` is unreliable in tiled kernels, so the flip-based form
    produces wrong gradients. The identity ``sum_{i>=j} g_i = total - sum_{i<j}
    g_i = total - cumsum(g)_j + g_j`` uses only cumsum/sum/sub/add (all
    correctly supported), giving the correct reverse cumulative sum. Fires only
    on the exact rev∘cumsum∘rev pattern, so other kernels are unaffected.
    """
    rev = torch.ops.prims.rev.default
    cumsum = torch.ops.aten.cumsum.default
    fired = False
    for outer in list(graph.nodes):
        if outer.op != "call_function" or outer.target is not rev:
            continue
        cs = outer.args[0]
        outer_dims = outer.args[1]
        if not (
            isinstance(cs, Node) and cs.op == "call_function" and cs.target is cumsum
        ):
            continue
        inner = cs.args[0]
        dim = cs.args[1]
        if not (
            isinstance(inner, Node)
            and inner.op == "call_function"
            and inner.target is rev
        ):
            continue
        if outer_dims != [dim] or inner.args[1] != [dim]:
            continue
        g = inner.args[0]
        with graph.inserting_before(outer):
            total = graph.call_function(
                torch.ops.aten.sum.dim_IntList, (g, [dim], True)
            )
            csum = graph.call_function(cumsum, (g, dim))
            sub = graph.call_function(torch.ops.aten.sub.Tensor, (total, csum))
            res = graph.call_function(torch.ops.aten.add.Tensor, (sub, g))
            for nn in (total, csum, sub, res):
                if outer.meta:
                    nn.meta = outer.meta.copy()
        outer.replace_all_uses_with(res)
        graph.erase_node(outer)
        fired = True
    return fired


def _rewrite_dtype_matmul(graph: torch.fx.Graph) -> None:
    """Rewrite out_dtype matmul overloads (``bmm.dtype`` / ``baddbmm.dtype`` /
    ``mm.dtype`` / ``addmm.dtype``) to their ``.default`` form with the matmul
    operands cast up to ``out_dtype``.

    A kernel that calls e.g. ``torch.bmm(a, b, torch.float32)`` (the way
    ``hl.dot(..., out_dtype=...)`` lowers) produces an ``aten.bmm.dtype`` node
    carrying ``out_dtype`` as a *positional* argument. torch's ``scan``
    higher-order-op re-traces the combine_fn via ``make_fx`` during autograd
    and cannot reconstruct that positional overload, raising
    ``aten::bmm() is missing value for argument 'out_dtype'``. The forward
    Helion lowering applies the same out_dtype via
    ``_apply_bmm_dot_dtype_requirements``; casting the operands and using
    ``.default`` is the faithful equivalent (accumulation happens in
    ``out_dtype``) and keeps the graph differentiable.
    """
    # out_dtype is the positional arg right after the tensor operands; beta /
    # alpha (addmm / baddbmm) are keyword-only and stay in kwargs untouched.
    dtype_overloads: dict[object, tuple[object, int]] = {}
    for packet, default_op, n_tensor in (
        (torch.ops.aten.bmm, torch.ops.aten.bmm.default, 2),
        (torch.ops.aten.mm, torch.ops.aten.mm.default, 2),
        (torch.ops.aten.baddbmm, torch.ops.aten.baddbmm.default, 3),
        (torch.ops.aten.addmm, torch.ops.aten.addmm.default, 3),
    ):
        dtype_ov = getattr(packet, "dtype", None)
        if dtype_ov is not None:
            dtype_overloads[dtype_ov] = (default_op, n_tensor)

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        entry = dtype_overloads.get(node.target)
        if entry is None:
            continue
        default_op, n_tensor = entry
        # out_dtype may be encoded positionally (args[n_tensor]) or as the
        # 'out_dtype' keyword; handle both.
        kwargs = dict(node.kwargs)
        if len(node.args) > n_tensor:
            out_dtype = node.args[n_tensor]
            tensor_args = list(node.args[:n_tensor])
            trailing = list(node.args[n_tensor + 1 :])
        elif "out_dtype" in kwargs:
            out_dtype = kwargs.pop("out_dtype")
            tensor_args = list(node.args[:n_tensor])
            trailing = list(node.args[n_tensor:])
        else:
            # Helion applies out_dtype as a codegen requirement
            # (_apply_bmm_dot_dtype_requirements), not an FX arg, so the
            # .dtype node may carry only its tensor operands. Recover the
            # output dtype from fake-tensor meta.
            fake = node.meta.get("val") if node.meta else None
            out_dtype = fake.dtype if isinstance(fake, torch.Tensor) else None
            tensor_args = list(node.args[:n_tensor])
            trailing = list(node.args[n_tensor:])
        with graph.inserting_before(node):
            if isinstance(out_dtype, torch.dtype):
                tensor_args = [
                    graph.call_function(
                        torch.ops.aten._to_copy.default, (a,), {"dtype": out_dtype}
                    )
                    if isinstance(a, Node)
                    else a
                    for a in tensor_args
                ]
            new_node = graph.call_function(
                default_op,  # pyrefly: ignore [bad-argument-type]
                tuple(tensor_args) + tuple(trailing),
                kwargs,
            )
            new_node.meta = node.meta.copy() if node.meta else {}
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)


def differentiate_graph(
    compute_graph: torch.fx.Graph,
    input_tensors: tuple[torch.Tensor, ...],
    combine_attrs: dict[str, torch.fx.GraphModule] | None = None,
    bw_module_holder: list | None = None,
) -> torch.fx.Graph:
    """Differentiate ``compute_graph`` using AOT Autograd with full
    recomputation. ``combine_attrs`` registers sub-GraphModules (one per
    scan_op combine_fn) on the outer GraphModule so ``get_attr``
    references resolve. Inputs already carrying ``requires_grad=True``
    are detached+cloned to remain leaves.
    """
    example_inputs = []
    for t in input_tensors:
        if t.requires_grad:
            example_inputs.append(t.detach().clone().requires_grad_(True))
        else:
            rg = t.dtype.is_floating_point or t.dtype.is_complex
            example_inputs.append(
                torch.empty(t.shape, dtype=t.dtype, device=t.device, requires_grad=rg)
            )

    backward_graph: torch.fx.Graph | None = None

    def bw_compiler(
        gm: torch.fx.GraphModule,
        _aot_example_inputs: list[torch.Tensor],
    ) -> torch.fx.GraphModule:
        nonlocal backward_graph
        if _rewrite_reverse_cumsum(gm.graph):
            # Drop the now-orphaned rev/cumsum nodes the rewrite left behind so
            # they aren't emitted as dead `torch.flip`/`torch.cumsum` lines.
            gm.graph.eliminate_dead_code()
        gm.recompile()
        backward_graph = gm.graph
        if bw_module_holder is not None:
            bw_module_holder.append(gm)
        return gm

    root = dict(combine_attrs) if combine_attrs else {}
    decomps = dict(select_decomp_table())
    # prims.convert_element_type has no autograd formula and adding it
    # to the decomp table causes an infinite recursion
    # (_to_copy ↔ convert_element_type cycle).  Instead, rewrite any
    # convert_element_type nodes in a COPY of the graph to aten._to_copy
    # before tracing so AOT autograd can differentiate through them.
    # Use a deep copy so the shared compute_graph (also referenced by
    # FXToHelionConverter.compute_graph) is not permanently mutated in-place.
    _rewrite_convert_element_type(compute_graph)
    # Normalize out_dtype matmul overloads (bmm.dtype/baddbmm.dtype/...) that
    # torch's scan re-trace cannot reconstruct (see _rewrite_dtype_matmul).
    _rewrite_dtype_matmul(compute_graph)
    # Also rewrite combine_fn subgraph GraphModules: they are separate graphs
    # not visited by the top-level rewrite above.
    for sub_gm in root.values() if root else []:
        if isinstance(sub_gm, torch.fx.GraphModule):
            _rewrite_convert_element_type(sub_gm.graph)
            _rewrite_dtype_matmul(sub_gm.graph)
            # The combine_fn GraphModule was already compiled in
            # _build_combine_fn; in-place graph edits only take effect after
            # recompile(), otherwise scan_op runs the stale compiled forward.
            sub_gm.recompile()
    with functorch_config.patch(activation_memory_budget=0):
        compiled = aot_module_simplified(
            torch.fx.GraphModule(root, compute_graph),
            example_inputs,
            fw_compiler=lambda gm, _: gm,  # type: ignore[arg-type]
            bw_compiler=bw_compiler,  # type: ignore[arg-type]
            decompositions=decomps,
            partition_fn=min_cut_rematerialization_partition,
        )

        example_out = compiled(*example_inputs)
        if isinstance(example_out, (list, tuple)):
            loss = sum(
                (o.sum() for o in example_out if isinstance(o, torch.Tensor)),
                torch.zeros((), device=example_inputs[0].device, requires_grad=True),
            )
        else:
            loss = example_out.sum()
        loss.backward()  # pyrefly: ignore [missing-attribute]

    if backward_graph is None:
        raise exc.AutodiffNotSupported(
            "AOT autograd returned no backward graph for this kernel"
        )
    return backward_graph


class FXToHelionConverter:
    """Converts backward FX graph to Helion kernel source code."""

    _REDUCTION_OPS: ClassVar[set[str]] = {"sum", "amax", "amin", "mean"}

    def __init__(
        self,
        backward_graph: torch.fx.Graph,
        input_mappings: list[InputMapping],
        input_tensors: tuple[torch.Tensor, ...],
        grad_out_shapes: tuple[tuple[int, ...], ...],
        forward_reduce_dims: tuple[int, ...] = (),
        forward_full_slice_dims: tuple[int, ...] = (),
        forward_untiled_inputs: frozenset[str] = frozenset(),
        has_root_matmul: bool = False,
        bw_module: torch.fx.GraphModule | None = None,
        init_specs: list | None = None,
        compute_graph: torch.fx.Graph | None = None,
    ) -> None:
        self.backward_graph = backward_graph
        self.input_mappings = input_mappings
        self.input_tensors = input_tensors
        self.grad_input_order = [m.tensor_name for m in input_mappings]
        self.bw_module = bw_module
        self.init_specs = init_specs or []
        self.compute_graph = compute_graph

        self.primal_to_name = {
            i + 1: m.tensor_name for i, m in enumerate(input_mappings)
        }

        self.tensor_shapes: dict[str, tuple[int, ...]] = {
            m.tensor_name: tuple(input_tensors[i].shape)
            for i, m in enumerate(input_mappings)
        }

        self.grad_out_shapes = grad_out_shapes
        self.num_grad_outs = len(grad_out_shapes)

        # Track the exact set of grad_out param names we synthesise so that
        # _param_shape / _map_param_to_iter_dims can distinguish them from
        # user tensors that happen to start with "grad_out_".
        if len(grad_out_shapes) == 1:
            self._grad_out_params: frozenset[str] = frozenset({"grad_out"})
        else:
            self._grad_out_params = frozenset(
                f"grad_out_{i}" for i in range(len(grad_out_shapes))
            )

        self.forward_reduce_dims = forward_reduce_dims
        self.forward_full_slice_dims = forward_full_slice_dims
        self.forward_untiled_inputs = forward_untiled_inputs
        self._has_root_matmul = has_root_matmul

    def _grad_out_index(self, param_name: str) -> int:
        """Return the grad_out index encoded in `param_name` (e.g.
        ``grad_out_3`` -> 3). Single-output kernels use the bare name
        ``grad_out`` which maps to index 0."""
        if param_name == "grad_out":
            return 0
        assert param_name.startswith("grad_out_")
        return int(param_name[len("grad_out_") :])

    @staticmethod
    def _grad_target(grad_name: str) -> str:
        """Input tensor name a ``grad_<name>`` output corresponds to.

        Strips exactly one ``grad_`` prefix (``grad_name`` is always built as
        ``f"grad_{input_name}"``), so it stays correct even for an input that is
        itself named ``grad_*``.
        """
        return grad_name[len("grad_") :]

    def _param_ndim(self, param_name: str) -> int:
        """Get the ndim for an input parameter (grad_out or tensor input)."""
        return len(self._param_shape(param_name))

    def _param_shape(self, param_name: str) -> tuple[int, ...]:
        """Get the shape for an input parameter."""
        # Only dispatch to grad_out_shapes for params we explicitly synthesised
        # (tracked in _grad_out_params).  A user tensor whose name happens to
        # start with "grad_out_" must NOT be treated as an upstream gradient
        # reference.
        if param_name in self._grad_out_params:
            return self.grad_out_shapes[self._grad_out_index(param_name)]
        return self.tensor_shapes[param_name]

    def _map_param_to_iter_dims(
        self, param_name: str, iter_shape: tuple[int, ...], non_reduced_dims: list[int]
    ) -> list[int]:
        """Map a parameter's dimensions to iter_shape dimension indices.

        Uses the known non-reduced dimensions to correctly match params that
        have fewer dims than iter_shape.

        Strategy:
        1. If this is a `grad_out` param with the right rank, its dims must
           map to the non-reduced dims (by definition — grad_out has the
           shape of the forward output).
        2. For other params, fall through to size-based matching via
           `_match_dims`, which surfaces ambiguity / unmatchable cases.
        """
        param_shape = self._param_shape(param_name)
        param_ndim = len(param_shape)
        iter_ndim = len(iter_shape)

        if param_ndim >= iter_ndim:
            return list(range(iter_ndim))

        if param_name in self._grad_out_params:
            if param_ndim == len(non_reduced_dims):
                return list(non_reduced_dims)

        return _match_dims(param_shape, iter_shape)

    def _iter_tensor_name(self) -> str:
        """Name of the input that spans the iteration shape (highest rank).

        Ties are broken deterministically by taking the first tensor in
        ``grad_input_order`` with the maximum rank (plain ``max()`` returns the
        *last* tied element, making the choice order-dependent).
        """
        best: str | None = None
        best_ndim = -1
        for name in self.grad_input_order:
            ndim = len(self.tensor_shapes[name])
            if ndim > best_ndim:
                best_ndim = ndim
                best = name
        assert best is not None
        return best

    def _has_reductions(self) -> bool:
        """Check if the backward graph contains reduction operations."""
        for node in self.backward_graph.nodes:
            if node.op == "call_function":
                op_name = getattr(node.target, "_opname", None)
                if op_name in self._REDUCTION_OPS:
                    return True
        return False

    def _has_cumulative_ops(self) -> bool:
        """Check for cumulative ops (``cumsum``/``cumprod``) in the backward.

        Like reductions, these fold along a dimension and so require that
        dimension to be kept whole (full-sliced), not tiled.
        """
        for node in self.backward_graph.nodes:
            if node.op == "call_function":
                op_name = getattr(node.target, "_opname", None)
                if op_name in ("cumsum", "cumprod"):
                    return True
        return False

    def _detect_reduced_dims(self, iter_shape: tuple[int, ...]) -> list[int]:
        """Return iter dims reduced from input to forward output.

        Prefers the forward IR's explicit reduce dims (authoritative). Falls
        back to shape-diffing iter_shape against grad_out; raises on ambiguity
        rather than guessing.

        For multi-output kernels, uses the most-reduced output (minimum ndim)
        to infer the reduction pattern.  A full-shape output at index 0 does
        not mean no reduction occurred — a later output may be reduced.
        """
        iter_ndim = len(iter_shape)
        if not self.grad_out_shapes:
            return []

        # Use the most-reduced grad_out (minimum ndim) to infer reduction dims.
        # If grad_out_shapes[0] is full-shape but grad_out_shapes[1] is reduced,
        # using only index 0 would miss the reduction entirely.
        grad_shape = min(self.grad_out_shapes, key=len)
        grad_ndim = len(grad_shape)

        if grad_ndim >= iter_ndim:
            return []

        if self.forward_reduce_dims:
            normalized = sorted({d % iter_ndim for d in self.forward_reduce_dims})
            # Only trust the forward IR when its reduction count matches the
            # input→output dim drop; otherwise the reductions are internal
            # (e.g. the mean inside RMS norm whose result is unsqueezed back
            # to full shape) and shape-diffing is the right fallback.
            if len(normalized) == iter_ndim - grad_ndim:
                return normalized

        def _find_reduced(grad_pos: int, iter_pos: int) -> list[list[int]]:
            if grad_pos == grad_ndim:
                return [list(range(iter_pos, iter_ndim))]
            if iter_pos == iter_ndim:
                return []
            results = []
            if iter_shape[iter_pos] == grad_shape[grad_pos]:
                results.extend(_find_reduced(grad_pos + 1, iter_pos + 1))
            for rest in _find_reduced(grad_pos, iter_pos + 1):
                results.append([iter_pos, *rest])
            return results

        all_reduced = _find_reduced(0, 0)

        if len(all_reduced) == 1:
            return all_reduced[0]
        if len(all_reduced) == 0:
            raise exc.AutodiffNotSupported(
                f"cannot align grad_out shape {grad_shape} with iter shape {iter_shape}"
            )
        raise exc.AutodiffNotSupported(
            f"ambiguous reduction inference: iter shape {iter_shape} could "
            f"reduce to {grad_shape} via dims {all_reduced}; pass an explicit "
            "reduction in the forward kernel"
        )

    def _get_backward_reduce_dims(self, iter_ndim: int) -> list[int]:
        """Reduce dim indices from backward graph reduction ops, normalized
        to non-negative via ``% iter_ndim``."""
        return sorted(_collect_reduce_dims(self.backward_graph, iter_ndim=iter_ndim))

    def convert(self) -> str:
        """Convert the backward FX graph to Helion kernel source code."""
        placeholders = []
        computations = []
        output_node = None

        for node in self.backward_graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)
            elif node.op == "call_function":
                computations.append(node)
            elif node.op == "output":
                output_node = node

        # Multi-tile-loop bw FX (has scan_op nodes) uses a separate lowering.
        if any(
            n.op == "call_function" and getattr(n.target, "__name__", "") == "scan"
            for n in self.backward_graph.nodes
        ):
            return _convert_scan_op_bw_to_helion(
                backward_graph=self.backward_graph,
                bw_module=getattr(self, "bw_module", None),
                input_mappings=self.input_mappings,
                input_tensors=self.input_tensors,
                init_specs=getattr(self, "init_specs", []),
                grad_out_shapes=self.grad_out_shapes,
                compute_graph=self.compute_graph,
            )

        if self.num_grad_outs == 1:
            grad_out_params = ["grad_out"]
        else:
            grad_out_params = [f"grad_out_{i}" for i in range(self.num_grad_outs)]
        input_params = [*grad_out_params, *self.grad_input_order]
        output_grad_names = [f"grad_{name}" for name in self.grad_input_order]

        # Reject tensor names that are Python keywords: they would be emitted
        # as function-parameter names in the generated kernel, which Python's
        # compiler rejects with a cryptic SyntaxError.
        for name in self.grad_input_order:
            if keyword.iskeyword(name):
                raise exc.AutodiffNotSupported(
                    f"tensor name {name!r} is a Python keyword and cannot be "
                    "used as a kernel parameter name in the generated backward "
                    "kernel; rename the tensor to avoid the conflict"
                )
            if not name.isidentifier():
                raise exc.AutodiffNotSupported(
                    f"tensor name {name!r} is not a valid Python identifier and "
                    "cannot be used as a kernel parameter name in the generated "
                    "backward kernel; rename the tensor"
                )

        self._backward_has_reductions = self._has_reductions()

        iter_tensor_name = self._iter_tensor_name()
        iter_shape = self.tensor_shapes[iter_tensor_name]
        iter_ndim = len(iter_shape)

        # Four related dim sets (easy to confuse):
        #   _forward_reduced_dims: dims missing from grad_out
        #   _full_slice_dims: dims indexed with ':' inside the tile loop
        #   _non_reduced_dims: complement of _forward_reduced_dims
        #   _tiled_dims: dims we iterate (= iter dims minus _full_slice_dims)
        # They diverge when the backward needs reductions but the forward
        # output is full-shape (e.g. softmax).
        self._forward_reduced_dims = self._detect_reduced_dims(iter_shape)

        if self._forward_reduced_dims:
            self._full_slice_dims = list(self._forward_reduced_dims)
        elif self._has_root_matmul and self.forward_full_slice_dims:
            # Single-loop matmul: the full-slice dim is the contraction dim taken
            # from the forward's load pattern. A backward reduction over the
            # *tiled* loop dim (e.g. a bias gradient summing over the rows) is
            # handled as a split-reduction across tiles, NOT a full-slice fold,
            # so it must not override the contraction dim here.
            self._full_slice_dims = [
                d for d in self.forward_full_slice_dims if d < iter_ndim
            ]
        elif (
            self.forward_full_slice_dims
            and self._backward_has_reductions
            and iter_ndim >= 2
        ):
            # The backward has reductions but the forward output is
            # full-shape (e.g. softmax, rms-norm).  The compute graph
            # uses full-tensor shapes, so the backward FX reductions
            # expect the FULL reduction dimension to be available.
            # Use the backward's reduction dims for full-slicing so the
            # generated kernel can fold the reductions correctly.
            # Fall back to the forward full-slice pattern only when it
            # already covers the backward's reduction dims.
            bwd_dims = self._get_backward_reduce_dims(iter_ndim)
            fwd_fs = {d for d in self.forward_full_slice_dims if d < iter_ndim}
            if bwd_dims and not set(bwd_dims).issubset(fwd_fs):
                if len(bwd_dims) >= iter_ndim:
                    self._full_slice_dims = [bwd_dims[-1]]
                else:
                    self._full_slice_dims = bwd_dims
            else:
                self._full_slice_dims = [
                    d for d in self.forward_full_slice_dims if d < iter_ndim
                ]
        elif self.forward_full_slice_dims:
            # Shape-preserving kernel without backward reductions: use the
            # forward's actual full-slice pattern from load/store subscripts.
            self._full_slice_dims = [
                d for d in self.forward_full_slice_dims if d < iter_ndim
            ]
        elif self._backward_has_reductions and iter_ndim >= 2:
            # Backward reduces but forward output is full-shape (e.g. softmax,
            # rms-norm's weight grad). When backward touches every dim, keep
            # the last as full-slice and let the rest tile via partial buffers.
            bwd_dims = self._get_backward_reduce_dims(iter_ndim)
            if not bwd_dims:
                raise exc.AutodiffNotSupported(
                    "backward reduction has no resolvable iter dim "
                    "(possibly a full-tensor reduction without an explicit dim)"
                )
            if len(bwd_dims) >= iter_ndim:
                self._full_slice_dims = [bwd_dims[-1]]
            else:
                self._full_slice_dims = bwd_dims
        else:
            self._full_slice_dims = []

        self._non_reduced_dims = [
            d for d in range(iter_ndim) if d not in self._forward_reduced_dims
        ]
        self._tiled_dims = [
            d for d in range(iter_ndim) if d not in self._full_slice_dims
        ]

        self._needs_broadcast = any(
            len(s) < iter_ndim for s in self.grad_out_shapes
        ) or any(len(self.tensor_shapes[p]) < iter_ndim for p in self.tensor_shapes)
        # Cumulative ops (cumsum) also fold along a dim and need the full-slice
        # path so that dim is kept whole rather than tiled.  A root-graph matmul
        # also needs it: the contraction dim must stay whole (full-slice) and the
        # weight gradient must accumulate across tiles via a partial buffer.
        self._use_reduction_kernel_path = (
            (
                self._backward_has_reductions
                or self._has_cumulative_ops()
                or self._has_root_matmul
            )
            and iter_ndim >= 2
            and len(self._full_slice_dims) > 0
        )

        # A single-loop matmul must be differentiated through the reduction path:
        # it tiles only the loop dim(s), keeps the contraction dim whole, loads
        # each operand per its forward tiling, and stores each grad per-tile (for
        # a tiled operand, e.g. batched bmm) or accumulated across tiles (for a
        # fully-loaded shared weight). If the reduction path can't be reached
        # (no full-slice/contraction dim), the elementwise/broadcast paths would
        # block-slice the operands and emit a silently wrong gradient — reject.
        # Also reject when the matmul output is itself reduced in the same loop
        # (`_forward_reduced_dims`): the reduced dim would take the full-slice
        # slot instead of the contraction dim, block-slicing the operands.
        if self._has_root_matmul and (
            not self._use_reduction_kernel_path or self._forward_reduced_dims
        ):
            raise exc.AutodiffNotSupported(
                "a matmul in a single (non-nested) `hl.tile` loop is only "
                "supported when its contraction dimension is loaded whole and its "
                "output is not further reduced in the same loop; this kernel's "
                "matmul does not fit that pattern, so a correct backward cannot "
                "be generated. Express the matmul as a nested `hl.tile` reduction "
                "over the contraction dimension instead."
            )

        computation_lines, node_to_var = self._generate_computation(
            computations, placeholders
        )
        output_assignments = self._generate_output_assignments(output_node, node_to_var)

        return self._build_source(
            input_params, output_grad_names, computation_lines, output_assignments
        )

    def _get_var_name(self, node_name: str) -> str:
        """Map backward graph node name to generated variable name.

        ``primals_N`` → original input tile, ``tangents_N`` → grad_out tile.
        """
        if node_name.startswith("primals_"):
            idx = int(node_name.split("_")[1])
            if idx not in self.primal_to_name:
                raise exc.AutodiffNotSupported(
                    f"backward graph references {node_name!r} but only "
                    f"{len(self.primal_to_name)} input mapping(s) were provided "
                    f"(keys: {sorted(self.primal_to_name)}); this can happen when "
                    "AOT Autograd adds extra primals via activation checkpointing "
                    "or re-materialization — rewrite the kernel without those features"
                )
            return f"{self.primal_to_name[idx]}_tile"
        if node_name.startswith("tangents_"):
            if self.num_grad_outs == 1:
                return "grad_out_tile"
            idx = int(node_name.split("_")[1]) - 1
            return f"grad_out_{idx}_tile"
        return f"{node_name}_val"

    def _generate_computation(
        self, computations: list[Node], placeholders: list[Node]
    ) -> tuple[list[str], dict[str, str]]:
        """Generate Python code for each computation node.

        With full recomputation (activation_memory_budget=0), the backward graph
        contains all forward computation ops. Placeholders are only primals_
        (original inputs) and tangents_ (upstream gradients).

        Returns:
            (computation_lines, node_to_var) where node_to_var maps node names
            to generated variable names (including aliases from skipped ops).
        """
        lines = []
        node_to_var: dict[str, str] = {}

        def process_arg(arg: object) -> str:
            if isinstance(arg, Node):
                return node_to_var[arg.name]
            if isinstance(arg, (list, tuple)):
                processed = [process_arg(item) for item in arg]
                return f"[{', '.join(processed)}]"
            return repr(arg)

        # Catch sparse tangents_N (e.g. dead outputs) — would mis-pair indices.
        tangent_indices: list[int] = []
        for ph in placeholders:
            node_to_var[ph.name] = self._get_var_name(ph.name)
            if ph.name.startswith("tangents_"):
                tangent_indices.append(int(ph.name.split("_")[1]))
        if tangent_indices:
            assert sorted(tangent_indices) == list(range(1, self.num_grad_outs + 1)), (
                f"expected dense tangents_1..{self.num_grad_outs}, got "
                f"{sorted(tangent_indices)}"
            )

        for node in computations:
            target = node.target
            op_name = getattr(target, "_opname", None)

            # Identity ops alias the input variable (no codegen).
            if op_name in {"detach", "alias"}:
                if node.args:
                    input_node = node.args[0]
                    assert isinstance(input_node, Node)
                    node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            # Inline scalar_tensor as a literal at use sites.
            if op_name == "scalar_tensor":
                if node.args:
                    node_to_var[node.name] = repr(node.args[0])
                    continue

            # _to_copy / convert_element_type → tensor.to(dtype)
            if op_name in ("_to_copy", "convert_element_type"):
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    input_var = node_to_var[input_node.name]
                    dtype_arg = node.kwargs.get("dtype")
                    if dtype_arg is None and len(node.args) >= 2:
                        dtype_arg = node.args[1]
                    if isinstance(dtype_arg, torch.dtype):
                        dtype_str = f"torch.{str(dtype_arg).split('.')[-1]}"
                        result_var = f"{node.name}_val"
                        node_to_var[node.name] = result_var
                        lines.append(f"{result_var} = {input_var}.to({dtype_str})")
                    else:
                        # _to_copy without dtype (layout/device/pin_memory only):
                        # alias the input since no semantic change is needed.
                        node_to_var[node.name] = input_var
                    continue

            if op_name == "unsqueeze":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    # In the non-reduction path, `tangents_N` tiles are already
                    # loaded with the correct shape (tile loading inserts `None`
                    # indices for missing dims), so aliasing is correct.  For
                    # non-tangent inputs (primals or intermediates) the tile
                    # variable has the same ndim as the original tensor and the
                    # unsqueeze must be emitted explicitly.
                    is_tangent_input = input_node.name.startswith("tangents_")
                    if self._use_reduction_kernel_path or not is_tangent_input:
                        result_var = f"{node.name}_val"
                        node_to_var[node.name] = result_var
                        input_var = node_to_var[input_node.name]
                        assert "val" in input_node.meta, (
                            f"unsqueeze input {input_node.name} missing fake-tensor meta"
                        )
                        in_ndim = input_node.meta["val"].ndim
                        dim = node.args[1] if len(node.args) > 1 else -1
                        if in_ndim <= 1:
                            out_ndim = in_ndim + 1
                            # in_ndim ≤ 1 ⇒ out_ndim ≤ 2, so `shape` has at
                            # most one -1 (set explicitly to 1 below).
                            assert out_ndim <= 2
                            if isinstance(dim, int):
                                if dim < 0:
                                    dim = out_ndim + dim
                                shape = [-1] * out_ndim
                                shape[dim] = 1
                            else:
                                shape = [-1, 1]
                            shape_str = ", ".join(str(d) for d in shape)
                            lines.append(
                                f"{result_var} = {input_var}.reshape({shape_str})"
                            )
                        else:
                            out_ndim = in_ndim + 1
                            if isinstance(dim, int):
                                if dim < 0:
                                    dim = out_ndim + dim
                            else:
                                dim = out_ndim - 1
                            idx = []
                            for i in range(out_ndim):
                                if i == dim:
                                    idx.append("None")
                                else:
                                    idx.append(":")
                            lines.append(
                                f"{result_var} = {input_var}[{', '.join(idx)}]"
                            )
                    else:
                        # Tangent input in non-reduction path: tile loading already
                        # applied the correct shape, so alias is sufficient.
                        node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            # `expand` aliases its input — Triton broadcasts at elementwise
            # ops, so AOT-decomposed graphs don't need explicit materialization.
            if op_name == "expand":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            if op_name == "view":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    if self._use_reduction_kernel_path and len(node.args) >= 2:
                        target_shape = node.args[1]
                        if isinstance(target_shape, (list, tuple)):
                            assert "val" in input_node.meta, (
                                f"view input {input_node.name} missing fake-tensor meta"
                            )
                            in_ndim = input_node.meta["val"].ndim
                            data_dims = sum(
                                1 for d in target_shape if isinstance(d, int) and d > 1
                            )
                            result_var = f"{node.name}_val"
                            input_var = node_to_var[input_node.name]
                            if data_dims <= 1:
                                dyn = [
                                    -1 if isinstance(d, int) and d > 1 else d
                                    for d in target_shape
                                ]
                                node_to_var[node.name] = result_var
                                shape_str = ", ".join(str(d) for d in dyn)
                                lines.append(
                                    f"{result_var} = {input_var}.reshape({shape_str})"
                                )
                            else:
                                ones = [
                                    i
                                    for i, d in enumerate(
                                        target_shape  # pyrefly: ignore [bad-argument-type]
                                    )
                                    if isinstance(d, int) and d == 1
                                ]
                                if ones and len(target_shape) == in_ndim + len(ones):
                                    idx = []
                                    for d in target_shape:
                                        if isinstance(d, int) and d == 1:
                                            idx.append("None")
                                        else:
                                            idx.append(":")
                                    node_to_var[node.name] = result_var
                                    lines.append(
                                        f"{result_var} = {input_var}[{', '.join(idx)}]"
                                    )
                                else:
                                    # Complex reshape (multiple non-trivial data
                                    # dims, not a pure unsqueeze) cannot be lowered
                                    # correctly in tile context — raise instead of
                                    # silently aliasing to wrong-shaped input.
                                    raise exc.AutodiffNotSupported(
                                        f"view{list(target_shape)} on {in_ndim}D input "
                                        f"cannot be lowered in tile context "
                                        f"(data_dims={data_dims}); "
                                        f"reshape to a single non-trivial dim or use a "
                                        f"pure unsqueeze instead"
                                    )
                            continue
                    node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            if op_name == "squeeze":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    if self._use_reduction_kernel_path:
                        result_var = f"{node.name}_val"
                        node_to_var[node.name] = result_var
                        input_var = node_to_var[input_node.name]
                        assert "val" in node.meta, (
                            f"squeeze node {node.name} missing fake-tensor meta"
                        )
                        out_ndim = node.meta["val"].ndim
                        if out_ndim <= 1:
                            lines.append(f"{result_var} = {input_var}.reshape(-1)")
                        elif len(node.args) > 1:
                            dim = node.args[1]
                            lines.append(f"{result_var} = {input_var}.squeeze({dim})")
                        else:
                            # Argless squeeze: derive dim from in/out shape diff.
                            assert "val" in input_node.meta
                            in_shape = tuple(input_node.meta["val"].shape)
                            out_shape = tuple(node.meta["val"].shape)
                            squeezed = _diff_squeezed_dims(in_shape, out_shape)
                            if squeezed is None or len(squeezed) != 1:
                                raise exc.AutodiffNotSupported(
                                    f"squeeze without dim cannot be lowered "
                                    f"unambiguously ({in_shape} -> {out_shape})"
                                )
                            lines.append(
                                f"{result_var} = {input_var}.squeeze({squeezed[0]})"
                            )
                    else:
                        node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            # Intercept factory ops with static shape args before generic dispatch,
            # which would inline e.g. `torch.full([32, 32], 0)` — a static size
            # incompatible with dynamic tile shapes.
            # Rewrite to *_like variants that infer shape from an existing tile.
            _FACTORY_LIKE_OPS = {
                "full",
                "zeros",
                "ones",
                "empty",
                "full_like",
                "zeros_like",
                "ones_like",
                "empty_like",
            }
            if op_name in _FACTORY_LIKE_OPS:
                # Grab any already-bound tile var to use as the shape reference.
                ref_var = next(iter(node_to_var.values()), None)
                result_var = f"{node.name}_val"
                node_to_var[node.name] = result_var
                # Build kwargs string (include dtype etc.) — strip
                # layout/pin_memory/device which Triton cannot accept.
                _STRIP_KWARGS = {"layout", "pin_memory", "device"}
                kwarg_strs = [
                    f"{k}={v!r}"
                    for k, v in node.kwargs.items()
                    if k not in _STRIP_KWARGS
                ]
                if (
                    op_name in ("full", "zeros", "ones", "empty")
                    and ref_var is not None
                ):
                    # Replace static shape with tile-aware *_like form.
                    like_op = op_name + "_like"
                    fill_args = []
                    if op_name == "full" and len(node.args) >= 2:
                        fill_args = [repr(node.args[1])]
                    all_args = ", ".join([ref_var, *fill_args, *kwarg_strs])
                    code = f"torch.{like_op}({all_args})"
                elif op_name in ("full_like", "zeros_like", "ones_like", "empty_like"):
                    # Already *_like: thread the kwargs and pass through.
                    arg_vars = [process_arg(arg) for arg in node.args]
                    all_args = ", ".join(arg_vars + kwarg_strs)
                    code = f"torch.{op_name}({all_args})"
                else:
                    # Fallback: no ref_var available; emit as-is (rare edge case).
                    arg_vars = [process_arg(arg) for arg in node.args]
                    all_args = ", ".join(arg_vars + kwarg_strs)
                    code = f"torch.{op_name}({all_args})"
                lines.append(f"{result_var} = {code}")
                continue

            result_var = f"{node.name}_val"
            node_to_var[node.name] = result_var

            # Thread node.kwargs through to generated code so that dtype,
            # memory_format, device, etc. are not silently dropped.
            arg_vars = [process_arg(arg) for arg in node.args]
            kwarg_strs = [f"{k}={v!r}" for k, v in node.kwargs.items()]
            all_args_str = ", ".join(arg_vars + kwarg_strs)

            if op_name is not None and arg_vars:
                if hasattr(torch, op_name):
                    code = f"torch.{op_name}({all_args_str})"
                else:
                    tensor = arg_vars[0]
                    method_args = ", ".join(arg_vars[1:] + kwarg_strs)
                    code = f"{tensor}.{op_name}({method_args})"
            elif op_name is not None:
                code = f"torch.{op_name}({all_args_str})"
            else:
                code = f"{node.target}({all_args_str})"

            lines.append(f"{result_var} = {code}")

        return lines, node_to_var

    def _generate_output_assignments(
        self, output_node: Node | None, node_to_var: dict[str, str]
    ) -> list[tuple[str, str]]:
        """Map backward outputs to ``grad_<input>`` assignments.

        AOT Autograd returns gradients in forward-input order. ``node_to_var``
        already includes aliases from skipped identity/view/expand ops.
        """
        if output_node is None:
            return []

        output_args = output_node.args[0]
        if isinstance(output_args, (list, tuple)):
            output_args_list = list(output_args)
        else:
            output_args_list = [output_args]

        n_outputs = len(output_args_list)
        n_inputs = len(self.grad_input_order)
        if n_outputs > n_inputs:
            raise exc.AutodiffNotSupported(
                f"backward graph returned {n_outputs} output(s) but only "
                f"{n_inputs} input mapping(s) were provided; this can happen "
                "when the backward graph includes auxiliary outputs beyond the "
                "per-input gradients — rewrite the kernel to remove them"
            )

        assignments = []
        for i, out_node in enumerate(output_args_list):
            grad_name = f"grad_{self.grad_input_order[i]}"
            # AOT Autograd emits None for non-differentiable inputs; skip
            # instead of asserting so other gradients are still generated.
            if out_node is None:
                continue
            if not isinstance(out_node, Node):
                raise exc.AutodiffNotSupported(
                    f"backward graph output {i} is {out_node!r} (expected an "
                    "FX Node or None); cannot generate gradient assignment"
                )
            var_name = node_to_var.get(out_node.name, self._get_var_name(out_node.name))
            assignments.append((grad_name, var_name))

        return assignments

    def _build_source(
        self,
        input_params: list[str],
        output_grad_names: list[str],
        computation_lines: list[str],
        output_assignments: list[tuple[str, str]],
    ) -> str:
        """Build the complete Helion kernel source code via AST."""

        iter_tensor_name = self._iter_tensor_name()
        iter_var = f"grad_{iter_tensor_name}"
        iter_shape = self.tensor_shapes[iter_tensor_name]
        iter_ndim = len(iter_shape)

        needs_broadcast = self._needs_broadcast

        def parse_expr(code: str) -> ast.expr:
            return ast.parse(code, mode="eval").body

        def parse_stmt(code: str) -> ast.stmt:
            return ast.parse(code, mode="exec").body[0]

        imports: list[ast.stmt] = [
            ast.Import(names=[ast.alias(name="torch", asname=None)]),
            ast.Import(names=[ast.alias(name="helion", asname=None)]),
            ast.ImportFrom(
                module="helion",
                names=[ast.alias(name="language", asname="hl")],
                level=0,
            ),
        ]

        tensor_annotation = parse_expr("torch.Tensor")
        func_args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=p, annotation=tensor_annotation) for p in input_params],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )

        multi_output = len(output_grad_names) > 1
        return_annotation = parse_expr(
            "tuple[torch.Tensor, ...]" if multi_output else "torch.Tensor"
        )

        body: list[ast.stmt] = [
            parse_stmt(f"{g} = torch.empty_like({g.removeprefix('grad_')})")
            for g in output_grad_names
        ]

        loop_body: list[ast.stmt] = []

        if self._use_reduction_kernel_path:
            # Iterate tiled dims; ':' along full-slice dims so reductions
            # in the backward body can fold across them.
            full_slice_dims = set(self._full_slice_dims)
            tiled_dims = self._tiled_dims
            n_tiled = len(tiled_dims)

            dim_vars = [f"_dim_{i}" for i in range(n_tiled)]
            tile_vars = [f"tile_{i}" for i in range(n_tiled)]

            # e.g. iter_shape=(8,16,32) with full_slice={1} unpacks to
            # "_dim_0, _, _dim_1 = grad_x.shape".
            shape_parts = []
            dim_idx = 0
            for d in range(iter_ndim):
                if d in full_slice_dims:
                    shape_parts.append("_")
                else:
                    shape_parts.append(dim_vars[dim_idx])
                    dim_idx += 1
            body.append(parse_stmt(f"{', '.join(shape_parts)} = {iter_var}.shape"))

            # Same example: full_indices = [tile_0, ':', tile_1].
            full_indices: list[str] = []
            tv_i = 0
            for d in range(iter_ndim):
                if d in full_slice_dims:
                    full_indices.append(":")
                else:
                    full_indices.append(tile_vars[tv_i])
                    tv_i += 1

            for p in input_params:
                tensor_ndim = self._param_ndim(p)
                if self._has_root_matmul and p in self.forward_untiled_inputs:
                    # Weight loaded fully in the forward (e.g. a matmul weight):
                    # load all of it so the contraction dim stays whole; its
                    # gradient accumulates across the tile loop (split-reduction).
                    load_expr = f"{p}[{', '.join([':'] * tensor_ndim)}]"
                elif tensor_ndim < iter_ndim:
                    param_iter_dims = self._map_param_to_iter_dims(
                        p, iter_shape, self._non_reduced_dims
                    )
                    mapped_set = set(param_iter_dims)

                    if all(d in full_slice_dims for d in param_iter_dims):
                        # One ':' per param dim — `p[:]`, `p[:, :]`, etc.
                        load_expr = f"{p}[{', '.join([':'] * len(param_iter_dims))}]"
                    else:
                        indices = []
                        for d in range(iter_ndim):
                            if d in full_slice_dims:
                                if d in mapped_set:
                                    indices.append(":")
                            elif d in mapped_set:
                                tv_idx = tiled_dims.index(d)
                                indices.append(tile_vars[tv_idx])
                            else:
                                indices.append("None")
                        # Empty would mean the all-full-slice branch was missed.
                        assert indices, (
                            f"no load indices for {p}: param_iter_dims="
                            f"{param_iter_dims}, full_slice_dims={full_slice_dims}"
                        )
                        load_expr = f"{p}[{', '.join(indices)}]"
                else:
                    load_expr = f"{p}[{', '.join(full_indices)}]"
                loop_body.append(parse_stmt(f"{p}_tile = {load_expr}"))

            for line in computation_lines:
                loop_body.append(parse_stmt(line))

            # Outputs spanning iter or tiled shape store directly; smaller
            # outputs need partial-buffer accumulation across tiles.
            tiled_shape = tuple(iter_shape[d] for d in tiled_dims)
            split_reduction_outputs: list[tuple[str, str, str]] = []
            normal_outputs: list[tuple[str, str]] = []
            for grad_name, var_name in output_assignments:
                out_tensor = self._grad_target(grad_name)
                out_shape = self.tensor_shapes.get(out_tensor, ())
                # A fully-loaded (shared) weight's grad reduces over the tile
                # loop, so it must accumulate even when its ndim equals iter_ndim
                # (e.g. a square [n, n] matmul weight). Anything else spanning the
                # iter/tiled shape stores directly; smaller outputs accumulate too.
                untiled_weight = (
                    self._has_root_matmul and out_tensor in self.forward_untiled_inputs
                )
                spans_iter = len(out_shape) == iter_ndim or out_shape == tiled_shape
                if len(out_shape) == 0 or (spans_iter and not untiled_weight):
                    normal_outputs.append((grad_name, var_name))
                else:
                    split_reduction_outputs.append(
                        (grad_name, var_name, f"{grad_name}_parts")
                    )

            use_block_size = len(split_reduction_outputs) > 0
            if use_block_size and n_tiled != 1:
                # Partial buffer is indexed by `tile_vars[0].id`; multiple
                # tiled dims would race on the same slot.
                raise exc.AutodiffNotSupported(
                    f"split-reduction with {n_tiled} tiled dims would race "
                    "on the partial buffer; only a single tiled dim is "
                    "supported"
                )
            block_var = "_block_0"
            if use_block_size:
                body.extend(
                    [
                        parse_stmt(
                            f"{block_var} = hl.register_block_size({dim_vars[0]})"
                        ),
                        parse_stmt(
                            f"_num_blocks = ({dim_vars[0]} + {block_var} - 1)"
                            f" // {block_var}"
                        ),
                    ]
                )
                for grad_name, _var_name, partial_name in split_reduction_outputs:
                    out_tensor = self._grad_target(grad_name)
                    out_ndim = len(self.tensor_shapes[out_tensor])
                    trailing_dims = ", ".join(
                        f"{out_tensor}.shape[{d}]" for d in range(out_ndim)
                    )
                    # Accumulate partials in (at least) float32 (cast back at the
                    # end) so a low-precision weight (fp16/bf16) doesn't lose
                    # precision or overflow across tiles, while fp64 stays fp64.
                    # `zeros` (not `empty`) so any slot the loop doesn't write
                    # contributes 0 rather than uninitialized garbage (which in
                    # fp16 is frequently a NaN bit pattern).
                    body.append(
                        parse_stmt(
                            f"{partial_name} = torch.zeros("
                            f"[_num_blocks, {trailing_dims}],"
                            f" dtype=torch.promote_types("
                            f"{out_tensor}.dtype, torch.float32),"
                            f" device={out_tensor}.device)"
                        )
                    )

            for grad_name, var_name in normal_outputs:
                out_tensor = self._grad_target(grad_name)
                out_ndim = len(self.tensor_shapes.get(out_tensor, ()))
                if out_ndim == iter_ndim:
                    store_idx = ", ".join(full_indices)
                elif out_ndim == n_tiled:
                    store_idx = ", ".join(tile_vars)
                else:
                    raise exc.AutodiffNotSupported(
                        f"grad output {grad_name} has ndim {out_ndim}; expected "
                        f"either {iter_ndim} (full iter) or {n_tiled} (tiled "
                        "dims) for the normal-store path"
                    )
                loop_body.append(parse_stmt(f"{grad_name}[{store_idx}] = {var_name}"))
            for grad_name, var_name, partial_name in split_reduction_outputs:
                # partial_buffer shape is `[_num_blocks, *out_shape]`; emit
                # one ':' per output dim so any rank works.
                out_tensor = self._grad_target(grad_name)
                out_ndim = len(self.tensor_shapes.get(out_tensor, ()))
                trailing = ", ".join([":"] * out_ndim)
                loop_body.append(
                    parse_stmt(
                        f"{partial_name}[{tile_vars[0]}.id, {trailing}] = {var_name}"
                    )
                )

            if n_tiled == 1:
                tile_target = ast.Name(id=tile_vars[0], ctx=ast.Store())
                if use_block_size:
                    tile_iter = parse_expr(
                        f"hl.tile({dim_vars[0]}, block_size={block_var})"
                    )
                else:
                    tile_iter = parse_expr(f"hl.tile({dim_vars[0]})")
            else:
                tile_target = ast.Tuple(
                    elts=[ast.Name(id=tv, ctx=ast.Store()) for tv in tile_vars],
                    ctx=ast.Store(),
                )
                tile_iter = parse_expr(f"hl.tile([{', '.join(dim_vars)}])")

            body.append(
                ast.For(
                    target=tile_target,
                    iter=tile_iter,
                    body=loop_body,
                    orelse=[],
                )
            )

            for grad_name, _var_name, partial_name in split_reduction_outputs:
                out_tensor = self._grad_target(grad_name)
                body.append(
                    parse_stmt(
                        f"{grad_name} = {partial_name}.sum(0).to({out_tensor}.dtype)"
                    )
                )
        elif needs_broadcast:
            dim_vars = [f"_dim_{i}" for i in range(iter_ndim)]
            tile_vars = [f"tile_{i}" for i in range(iter_ndim)]

            if iter_ndim > 1:
                body.append(parse_stmt(f"{', '.join(dim_vars)} = {iter_var}.shape"))
            else:
                body.append(parse_stmt(f"({dim_vars[0]},) = {iter_var}.shape"))

            for p in input_params:
                tensor_ndim = self._param_ndim(p)
                if tensor_ndim < iter_ndim:
                    dim_map = self._map_param_to_iter_dims(
                        p, iter_shape, self._non_reduced_dims
                    )
                    mapped_iter_dims = set(dim_map)
                    indices = []
                    for i in range(iter_ndim):
                        if i in mapped_iter_dims:
                            indices.append(tile_vars[i])
                        else:
                            indices.append("None")
                    load_expr = f"{p}[{', '.join(indices)}]"
                else:
                    load_expr = f"{p}[{', '.join(tile_vars)}]"
                loop_body.append(parse_stmt(f"{p}_tile = {load_expr}"))

            for line in computation_lines:
                loop_body.append(parse_stmt(line))

            # Use rank-appropriate tile_vars per output, not the full tile_vars
            # for every output regardless of its rank.
            for grad_name, var_name in output_assignments:
                out_tensor = self._grad_target(grad_name)
                out_shape = self.tensor_shapes.get(out_tensor, ())
                out_ndim = len(out_shape)
                if out_ndim >= iter_ndim:
                    # Full-rank output: index with all tile_vars.
                    store_idx = ", ".join(tile_vars)
                else:
                    # Lower-rank output: select only the tile vars whose iter
                    # dim maps to one of the output's dimensions.
                    try:
                        dim_map = self._map_param_to_iter_dims(
                            out_tensor, iter_shape, self._non_reduced_dims
                        )
                    except exc.AutodiffNotSupported:
                        dim_map = list(range(out_ndim))
                    mapped_set = set(dim_map)
                    store_idx = ", ".join(
                        tile_vars[i] for i in range(iter_ndim) if i in mapped_set
                    )
                    if not store_idx:
                        store_idx = ", ".join(tile_vars[:out_ndim])
                loop_body.append(parse_stmt(f"{grad_name}[{store_idx}] = {var_name}"))

            tile_target = (
                ast.Name(id=tile_vars[0], ctx=ast.Store())
                if iter_ndim == 1
                else ast.Tuple(
                    elts=[ast.Name(id=tv, ctx=ast.Store()) for tv in tile_vars],
                    ctx=ast.Store(),
                )
            )
            body.append(
                ast.For(
                    target=tile_target,
                    iter=parse_expr(f"hl.tile([{', '.join(dim_vars)}])"),
                    body=loop_body,
                    orelse=[],
                )
            )
        else:
            # All inputs have iter_ndim — single shared `tile` indexer.
            for p in input_params:
                loop_body.append(parse_stmt(f"{p}_tile = {p}[tile]"))

            for line in computation_lines:
                loop_body.append(parse_stmt(line))

            for grad_name, var_name in output_assignments:
                loop_body.append(parse_stmt(f"{grad_name}[tile] = {var_name}"))

            body.append(
                ast.For(
                    target=ast.Name(id="tile", ctx=ast.Store()),
                    iter=parse_expr(f"hl.tile({iter_var}.shape)"),
                    body=loop_body,
                    orelse=[],
                )
            )

        if multi_output:
            return_value = ast.Tuple(
                elts=[ast.Name(id=g, ctx=ast.Load()) for g in output_grad_names],
                ctx=ast.Load(),
            )
        else:
            return_value = ast.Name(id=output_grad_names[0], ctx=ast.Load())
        body.append(ast.Return(value=return_value))

        func_def = ast.FunctionDef(
            name="backward_kernel",
            args=func_args,
            body=body,
            decorator_list=[parse_expr("helion.kernel()")],
            returns=return_annotation,
        )

        module = ast.Module(body=[*imports, func_def], type_ignores=[])
        ast.fix_missing_locations(module)

        source = ast.unparse(module)
        header = '"""\nAuto-generated Helion backward kernel.\n"""\n\n'
        return header + source


def _match_dims(param_shape: tuple[int, ...], iter_shape: tuple[int, ...]) -> list[int]:
    """Map each param dim to a unique iter dim by size.

    E.g. param ``[8, 32]`` with iter ``[8, 16, 32]`` → ``[0, 2]``. Raises
    ``AutodiffNotSupported`` when ambiguous (multiple iter dims match) or
    unmatchable. A silent fallback here previously produced wrong indexing.
    """
    param_ndim = len(param_shape)
    iter_ndim = len(iter_shape)

    if param_ndim >= iter_ndim:
        return list(range(iter_ndim))

    mapping: list[int] = []
    used: set[int] = set()
    for p_dim, p_size in enumerate(param_shape):
        candidates = [
            i for i, s in enumerate(iter_shape) if s == p_size and i not in used
        ]
        if not candidates:
            raise exc.AutodiffNotSupported(
                f"cannot map param dim {p_dim} (size {p_size}) into iter "
                f"shape {iter_shape}"
            )
        if len(candidates) > 1:
            raise exc.AutodiffNotSupported(
                f"ambiguous param→iter mapping: param shape {param_shape} "
                f"has size {p_size} which matches multiple iter dims "
                f"{candidates} in {iter_shape}"
            )
        mapping.append(candidates[0])
        used.add(candidates[0])

    return mapping


def _resolve_symnode_expr(
    sym_name: str,
    scalar_values: dict[str, object],
) -> object | None:
    """Resolve a symnode name to a concrete value.

    Handles both direct lookups (``sym_name in scalar_values``) and simple
    derived expressions like ``"1.0 - zuf0"`` by evaluating them with
    known scalars substituted.
    """
    if sym_name in scalar_values:
        return scalar_values[sym_name]
    # Try evaluating as a Python expression with known scalars.
    # Only allow safe builtins (no imports, no side-effects).
    try:
        return eval(sym_name, {"__builtins__": {}}, dict(scalar_values))
    except Exception:
        return None


def _resolve_scalar_values(
    kernel: Kernel[object],
    inputs: tuple[object, ...],
    host_function: HostFunction,
) -> dict[str, object]:
    """Map `_get_symnode` names (auto-generated, e.g. ``zuf0``) to concrete
    scalar arguments. Uses the current call's host_function so the symnode
    names match the specialization being differentiated.
    """
    all_args = kernel.normalize_args(*inputs)

    sig = inspect.signature(kernel.fn)
    param_names = list(sig.parameters.keys())
    param_to_value: dict[str, object] = {}
    for i, name in enumerate(param_names):
        if i < len(all_args) and not isinstance(all_args[i], torch.Tensor):
            param_to_value[name] = all_args[i]

    scalar_values: dict[str, object] = {}
    for param_name, fake_val in host_function.params.arguments.items():
        if param_name in param_to_value and not isinstance(fake_val, torch.Tensor):
            sym_name = str(fake_val)
            scalar_values[sym_name] = param_to_value[param_name]

    return scalar_values


def _collect_reduce_dims(
    graph: torch.fx.Graph,
    *,
    iter_ndim: int | None = None,
    skip_keepdim: bool = False,
) -> set[int]:
    """Gather ``dim`` args from sum/amax/amin/mean nodes in ``graph``.

    Reads positional or keyword ``dim`` arguments. When ``iter_ndim`` is
    given, normalizes negative dims to non-negative via ``% iter_ndim``.

    When ``skip_keepdim`` is True, reductions with ``keepdim=True`` are
    excluded.  These are internal intermediate computations (e.g. a running
    max in a log-sum-exp kernel) that do not change the output rank and
    should not count as forward output reduction dims.
    """
    reduce_dims: set[int] = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        op_name = getattr(node.target, "_opname", None)
        if op_name not in FXToHelionConverter._REDUCTION_OPS:
            continue
        # Skip keepdim=True reductions when requested (they are intermediate
        # computations that don't reduce the output rank).
        if skip_keepdim:
            keepdim = node.kwargs.get("keepdim", False)
            if not keepdim and len(node.args) >= 3:
                keepdim = node.args[2]
            if keepdim:
                continue
        dim_arg: object = None
        if len(node.args) >= 2:
            dim_arg = node.args[1]
        elif "dim" in node.kwargs:
            dim_arg = node.kwargs["dim"]
        if dim_arg is None:
            continue
        candidates = dim_arg if isinstance(dim_arg, (list, tuple)) else (dim_arg,)
        for d in candidates:
            if isinstance(d, int):
                reduce_dims.add(d % iter_ndim if iter_ndim is not None else d)
    return reduce_dims


def _extract_forward_full_slice_dims(forward_graph: torch.fx.Graph) -> tuple[int, ...]:
    """Detect dims that use full-slice (':') in the forward's load/store subscripts.

    Only considers tensors whose subscript ndim matches the max ndim seen
    (to avoid 1D weight[:] polluting 2D iteration dims).

    Returns the INTERSECTION of full-slice dims across all max-ndim subscripts:
    a dim is "full-sliced" only if it is ':' in EVERY max-ndim subscript.
    Using a union (any subscript) over-estimates full-slice dims and can make
    _tiled_dims empty, which would break backward kernel generation.
    """
    # First pass: find max subscript ndim
    max_ndim = 0
    for node in forward_graph.nodes:
        if node.op != "call_function":
            continue
        tn = getattr(node.target, "__name__", "")
        if tn not in ("load", "store"):
            continue
        sub = node.args[1]
        if isinstance(sub, (list, tuple)):
            max_ndim = max(max_ndim, len(sub))
    if max_ndim == 0:
        return ()
    # Second pass: compute INTERSECTION of full-slice dims from max-ndim subscripts.
    # Start with "all dims are full-slice" and narrow down.
    full_slice_dims: set[int] | None = None
    for node in forward_graph.nodes:
        if node.op != "call_function":
            continue
        tn = getattr(node.target, "__name__", "")
        if tn not in ("load", "store"):
            continue
        sub = node.args[1]
        if not isinstance(sub, (list, tuple)) or len(sub) != max_ndim:
            continue
        this_full = {
            d
            for d, idx in enumerate(sub)
            if isinstance(idx, slice) and idx == slice(None, None, None)
        }
        if full_slice_dims is None:
            full_slice_dims = this_full
        else:
            full_slice_dims &= this_full
    return tuple(sorted(full_slice_dims)) if full_slice_dims else ()


def _collect_store_reduce_dims(
    forward_graph: torch.fx.Graph,
) -> dict[str, frozenset[int]]:
    """Return a mapping from store-target-name → frozenset of (raw) reduction dims.

    For each ``store`` node in the graph, walks backward through its stored
    value to find any rank-reducing (keepdim=False) reduction ops that feed
    into it.  Reductions shared by multiple stores are attributed to each.

    The returned dims are raw (unnormalized) as in the graph's ``dim`` args.
    """
    _REDUCTION_OPS = FXToHelionConverter._REDUCTION_OPS

    def _get_node_name(node: torch.fx.Node) -> str:
        """Best-effort name for the tensor being stored."""
        host_tensor = node.args[0]
        if isinstance(host_tensor, torch.fx.Node):
            return host_tensor.name
        return node.name

    def _upstream_output_reduce_dims(
        node: torch.fx.Node, visited: set[int]
    ) -> frozenset[int]:
        """Recursively collect rank-reducing reduction dims upstream of ``node``."""
        if id(node) in visited:
            return frozenset()
        visited.add(id(node))
        dims: set[int] = set()
        op_name = (
            getattr(node.target, "_opname", None)
            if node.op == "call_function"
            else None
        )
        if op_name in _REDUCTION_OPS:
            # Skip keepdim=True: those don't change the output rank.
            keepdim = node.kwargs.get("keepdim", False)
            if not keepdim and len(node.args) >= 3:
                keepdim = node.args[2]
            if not keepdim:
                dim_arg: object = None
                if len(node.args) >= 2:
                    dim_arg = node.args[1]
                elif "dim" in node.kwargs:
                    dim_arg = node.kwargs["dim"]
                if dim_arg is not None:
                    candidates = (
                        dim_arg if isinstance(dim_arg, (list, tuple)) else (dim_arg,)
                    )
                    for d in candidates:
                        if isinstance(d, int):
                            dims.add(d)
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                dims |= _upstream_output_reduce_dims(arg, visited)
        return frozenset(dims)

    store_dims: dict[str, frozenset[int]] = {}
    for node in forward_graph.nodes:
        if node.op != "call_function":
            continue
        if getattr(node.target, "__name__", "") != "store":
            continue
        stored_value = node.args[2]
        if isinstance(stored_value, torch.fx.Node):
            dims = _upstream_output_reduce_dims(stored_value, set())
        else:
            dims = frozenset()
        store_dims[_get_node_name(node)] = dims
    return store_dims


def _extract_forward_reduce_dims(forward_graph: torch.fx.Graph) -> tuple[int, ...]:
    """Forward graph reduce dims, in iter-tile coordinates.

    The forward load tile has the same ndim as the iteration shape, so the
    raw ``dim`` args map onto iter dims with at most a sign flip. The
    consumer (`_detect_reduced_dims`) normalizes via ``% iter_ndim`` once
    iter_ndim is known.

    Reductions with ``keepdim=True`` are excluded: they are internal
    intermediate computations (e.g. a running max for numerical stability in
    log-sum-exp) that do not reduce the output rank and should not contribute
    to ``forward_reduce_dims``.

    If multiple output stores exist and they reduce DIFFERENT dimensions,
    raises ``AutodiffNotSupported`` because the backward would need per-output
    reduction dim tracking, which is not yet implemented.
    """
    store_dims = _collect_store_reduce_dims(forward_graph)

    if not store_dims:
        # No stores → fall back to the full-graph scan (handles edge cases).
        return tuple(sorted(_collect_reduce_dims(forward_graph, skip_keepdim=True)))

    # Separate stores that perform reductions from elementwise stores.
    # An elementwise store (empty frozenset) is fine alongside a reduction
    # store; we only care about the reduction stores' dims.
    reduction_store_dims = {name: dims for name, dims in store_dims.items() if dims}

    if not reduction_store_dims:
        # No stores have upstream reductions → no forward reduce dims.
        return ()

    # Check if all reduction stores agree on the same reduction dims.
    unique_dim_sets = set(reduction_store_dims.values())
    if len(unique_dim_sets) > 1:
        raise exc.AutodiffNotSupported(
            "multi-output kernel has outputs reducing different dimensions "
            f"({dict(reduction_store_dims)!r}); per-output reduction tracking is not "
            "yet supported — rewrite as separate kernels or use a single reduction axis"
        )

    # All reduction stores agree: return the common reduction dims.
    common_dims = next(iter(unique_dim_sets))
    return tuple(sorted(common_dims))


def _validate_no_subscript_permutation(graphs: list) -> None:
    """Reject kernels whose load/store subscripts index the *same set* of tile
    dimensions in *different* orders (e.g. a transposed load ``y[tn, tm]``).

    ``extract_computation_graph`` maps each load to a bare placeholder and
    discards the subscript order, so a permuted (transposed) load would
    silently differentiate the un-permuted expression and return a WRONG
    gradient. Detect the permutation up-front and raise rather than return a
    wrong answer. Matmul-style loads whose operands use *different* tile-dim
    sets (e.g. ``a[m, k]`` vs ``b[k, n]``) are not flagged.
    """
    for gi in graphs:
        graph = getattr(gi, "graph", None)
        if graph is None:
            continue
        # frozenset(block_size names) -> {arrangement: example node}
        by_set: dict[frozenset, dict[tuple, Node]] = {}
        for n in graph.nodes:
            if n.op != "call_function":
                continue
            if getattr(n.target, "__name__", "") not in ("load", "store"):
                continue
            sub = n.args[1]
            if not isinstance(sub, (list, tuple)):
                continue
            pos_to_name: dict[int, str] = {}
            for d, idx in enumerate(sub):
                if (
                    isinstance(idx, Node)
                    and getattr(idx.target, "__name__", "") == "_get_symnode"
                    and isinstance(idx.args[0], str)
                    and idx.args[0].startswith("block_size_")
                ):
                    pos_to_name[d] = idx.args[0]
            # Need >=2 distinct tiled dims for a permutation to exist.
            if len(set(pos_to_name.values())) < 2:
                continue
            key = frozenset(pos_to_name.values())
            arrangement = tuple(sorted((nm, d) for d, nm in pos_to_name.items()))
            seen = by_set.setdefault(key, {})
            seen[arrangement] = n
            if len(seen) > 1:
                raise exc.AutodiffNotSupported(
                    "load/store subscript permutation (e.g. a transposed tile "
                    f"index like y[tn, tm]) is not supported: the same tile dims "
                    f"{sorted(key)} are indexed in different orders, which would "
                    "silently produce a wrong gradient. Use an explicit "
                    "transpose op instead of a permuted subscript."
                )


# Matmul-family op base names (overload stripped, e.g. ``mm.default`` -> ``mm``).
_MATMUL_OPNAMES = frozenset({"mm", "bmm", "addmm", "baddbmm", "matmul", "dot"})


def _forward_has_root_matmul(forward_graph: torch.fx.Graph) -> bool:
    """True if a matmul sits directly in the *root* (outer) tile-loop body.

    A matmul inside a *nested* ``hl.tile`` loop lives in a separate subgraph and
    is differentiated through the scan path. A matmul in the *root* graph — i.e.
    a single ``hl.tile`` loop whose body contains ``x[tile_m, :] @ w`` — instead
    reaches the non-scan backward path, which must tile only the loop dim, load
    the matmul weight fully, and accumulate the weight gradient across tiles (see
    ``_use_reduction_kernel_path`` / ``forward_untiled_inputs``). Detecting it
    lets ``convert()`` route such kernels through that reduction path rather than
    the elementwise simple path (which would block-slice the weight and emit a
    numerically wrong gradient).
    """
    for n in forward_graph.nodes:
        if n.op != "call_function":
            continue
        if getattr(n.target, "__name__", "").split(".")[0] in _MATMUL_OPNAMES:
            return True
    return False


def _extract_forward_untiled_inputs(forward_graph: torch.fx.Graph) -> frozenset[str]:
    """Input tensors that the forward loads *fully* — i.e. with no tiled
    (``block_size``) index in any of their load subscripts.

    Such a tensor (e.g. a matmul weight ``w[:, :]``) is shared across every
    iteration of the tile loop, so its gradient is a *reduction over the loop*
    and must be accumulated (via a split-reduction partial buffer), not stored
    per-tile. A tensor that is ever indexed by a ``block_size`` symnode (e.g.
    ``x[tile_m, :]``) is tiled and its gradient is per-tile.
    """

    def _is_tiled_index(idx: object) -> bool:
        return (
            isinstance(idx, Node)
            and getattr(idx.target, "__name__", "") == "_get_symnode"
            and isinstance(idx.args[0], str)
            and idx.args[0].startswith("block_size_")
        )

    has_tiled: dict[str, bool] = {}
    for n in forward_graph.nodes:
        if n.op != "call_function" or getattr(n.target, "__name__", "") != "load":
            continue
        # The logical tensor name is the `_host_tensor` op's first arg, not the
        # FX node name (which can diverge via dedup suffixes / attribute origins
        # and then not match grad_input_order / forward_untiled_inputs lookups).
        host = n.args[0]
        if not (
            isinstance(host, Node)
            and getattr(host.target, "__name__", "") == "_host_tensor"
            and isinstance(host.args[0], str)
        ):
            continue
        name = host.args[0]
        sub = n.args[1]
        sub_list = sub if isinstance(sub, (list, tuple)) else [sub]
        tiled = any(_is_tiled_index(idx) for idx in sub_list)
        has_tiled[name] = has_tiled.get(name, False) or tiled
    return frozenset(name for name, tiled in has_tiled.items() if not tiled)


def _validate_supported_reductions(forward_graph: torch.fx.Graph) -> None:
    """Reject reductions outside `_REDUCTION_OPS`.

    Their backward decompositions emit ops we don't codegen for (e.g.
    ``aten.slice.Tensor`` from ``prod``) and would otherwise fail late with
    cryptic errors.

    Uses an allowlist: any op whose ``_opname`` is set (i.e., it is an ATen
    op) and is NOT in ``_REDUCTION_OPS`` and has a ``dim`` argument (making it
    a rank-reducing call) is rejected.  Common unsupported ops get a specific
    error message; others get a generic one.

    ``var`` and ``std`` are lowered to ``_inductor_lowering_extra`` nodes
    before this validator runs, so their ``_opname`` is never seen.  We detect
    them by checking the ``meta["orig_node"].target`` on those nodes.
    """
    _unsupported_with_message = {
        "prod",
        "var",
        "std",
        "argmax",
        "argmin",
        "any",
        "all",
        "cumsum",
        "cumprod",
    }
    # ATen ops that correspond to var/std and are lowered via
    # _inductor_lowering_extra before the validator sees them.
    _var_std_aten_ops = {
        torch.ops.aten.var.correction,
        torch.ops.aten.var_mean.correction,
        torch.ops.aten.std.correction,
        torch.ops.aten.std_mean.correction,
    }
    from helion.language._tracing_ops import _inductor_lowering_extra as _ile

    for node in forward_graph.nodes:
        if node.op != "call_function":
            continue

        # Detect var/std that were lowered to _inductor_lowering_extra before
        # the normal _opname check gets a chance to see them.
        if node.target is _ile:
            orig_node = node.meta.get("orig_node")
            if orig_node is not None and orig_node.target in _var_std_aten_ops:
                orig_op_name = getattr(orig_node.target, "_opname", "var/std")
                raise exc.AutodiffNotSupported(
                    f"reduction op {orig_op_name!r} (only "
                    f"{sorted(FXToHelionConverter._REDUCTION_OPS)} are supported)"
                )
            continue

        op_name = getattr(node.target, "_opname", None)
        if op_name is None:
            continue
        if op_name in FXToHelionConverter._REDUCTION_OPS:
            continue
        # Give an informative error for known-unsupported ops.
        if op_name in _unsupported_with_message:
            raise exc.AutodiffNotSupported(
                f"reduction op {op_name!r} (only "
                f"{sorted(FXToHelionConverter._REDUCTION_OPS)} are supported)"
            )
        # Catch-all: any op with an explicit keyword 'dim' argument that is not
        # in the supported set is an unsupported reduction.  Using the keyword
        # form (rather than positional arg[1]) avoids false-positives on binary
        # ops like 'div(x, y)' where args[1] is the divisor, not a dim.
        if "dim" in node.kwargs:
            raise exc.AutodiffNotSupported(
                f"reduction op {op_name!r} is not supported (only "
                f"{sorted(FXToHelionConverter._REDUCTION_OPS)} are supported)"
            )


def _extract_return_order(host_function: HostFunction) -> tuple[str, ...] | None:
    """Return host-tensor names in the order the kernel returns them.

    The user's ``grad_outs`` are in return order, but the forward graph stores
    values in first-store order — pairing them up needs the return order.
    Supports simple ``Name`` returns and ``expr.method(...)`` chains
    (e.g. ``inv_rms.reshape(-1, 1)``); returns None when the return value
    can't be reduced to plain names so callers fall back to store order.

    Also handles host-level post-loop reductions: if a returned name like
    ``final_loss`` is assigned as ``final_loss = torch.sum(loss)`` (or any
    call whose first positional argument is a plain name), it is resolved back
    to the stored tensor name (``loss``).

    Only the kernel's own ``Return`` statements are considered; returns
    inside nested functions or lambdas are skipped.
    """
    return_stmt = _find_outer_return(host_function.body)
    if return_stmt is None or return_stmt.value is None:
        return None

    value = return_stmt.value
    if isinstance(value, ast.Tuple):
        elements: list[ast.expr] = list(value.elts)
    else:
        elements = [value]

    # Build an assignment map from the body (recursing into if/for blocks):
    # var_name → rhs_expr.  This lets us trace ``final = f(src)`` even when
    # the assignment is inside a conditional branch.
    assign_map: dict[str, ast.expr] = {}

    def _collect_assigns(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                assign_map[stmt.targets[0].id] = stmt.value
            elif isinstance(stmt, ast.If):
                _collect_assigns(stmt.body)
                _collect_assigns(stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.With)):
                _collect_assigns(stmt.body)  # type: ignore[attr-defined]

    _collect_assigns(host_function.body)

    # Known reduction function names that may wrap a stored tensor in a
    # host-level post-loop computation (e.g. ``final = torch.sum(loss) / BT``).
    _REDUCTION_NAMES: frozenset[str] = frozenset(
        {"sum", "mean", "max", "min", "prod", "all", "any", "norm", "logsumexp"}
    )

    def _resolve_to_stored_name(name: str) -> str:
        """Follow assignment chain to find the original stored tensor name.

        Only traces through *reduction* calls (torch.sum, tensor.mean, etc.)
        and simple aliases (``x = y``).  Tensor-creation ops like
        ``torch.empty_like`` are not followed — they break the chain.
        """

        def _reduction_tensor_arg(node: ast.expr) -> str | None:
            """If ``node`` is a reduction call (or BinOp of one) applied to a
            plain Name, return that name; otherwise return None."""
            if isinstance(node, ast.Name) and node.id not in _MODULE_NAMES:
                return node.id
            if isinstance(node, ast.BinOp):
                # e.g. torch.sum(loss) / BT — try left first
                r = _reduction_tensor_arg(node.left)
                if r is not None:
                    return r
                return _reduction_tensor_arg(node.right)
            if isinstance(node, ast.Call):
                # Determine if this is a reduction (torch.sum, loss.mean(), etc.)
                func = node.func
                func_name: str | None = None
                if isinstance(func, ast.Attribute):
                    func_name = func.attr
                elif isinstance(func, ast.Name):
                    func_name = func.id
                if func_name not in _REDUCTION_NAMES:
                    return None  # not a reduction — stop tracing
                # First positional arg should be the tensor being reduced
                for a in node.args:
                    if isinstance(a, ast.Name) and a.id not in _MODULE_NAMES:
                        return a.id
                # loss.mean() pattern: the tensor is the callee object
                if isinstance(func, ast.Attribute):
                    return _reduction_tensor_arg(func.value)
            return None

        seen: set[str] = {name}
        while name in assign_map:
            rhs = assign_map[name]
            # Simple alias: ``final = loss``
            if isinstance(rhs, ast.Name) and rhs.id not in _MODULE_NAMES:
                candidate = rhs.id
            else:
                candidate = _reduction_tensor_arg(rhs)
            if candidate is None or candidate in seen:
                break
            seen.add(candidate)
            name = candidate
        return name

    names: list[str] = []
    for elt in elements:
        name = _underlying_name(elt)
        if name is None:
            return None
        # Resolve post-loop host reductions (e.g. final_loss → loss).
        name = _resolve_to_stored_name(name)
        names.append(name)
    return tuple(names)


def _find_outer_return(body: list[ast.stmt]) -> ast.Return | None:
    """First ``Return`` reachable from ``body`` without crossing a nested
    function or lambda boundary."""
    for stmt in body:
        if isinstance(stmt, ast.Return):
            return stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            inner = _find_outer_return_in_node(child)
            if inner is not None:
                return inner
    return None


def _find_outer_return_in_node(node: ast.AST) -> ast.Return | None:
    if isinstance(node, ast.Return):
        return node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return None
    for child in ast.iter_child_nodes(node):
        result = _find_outer_return_in_node(child)
        if result is not None:
            return result
    return None


def _assert_all_mapped(
    node: Node,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    node_map: dict[Node, Node],
) -> None:
    """Raise if any FX Node consumed by ``node`` is missing from ``node_map``.

    ``node_map.get`` returns ``None`` for unmapped Nodes, which would silently
    inject ``None`` into the rebuilt compute graph. The most common cause is
    a ``_get_symnode`` flowing into a real op without a resolved scalar value.
    """

    def visit(value: object) -> None:
        if isinstance(value, Node) and value not in node_map:
            raise exc.AutodiffNotSupported(
                f"node {node.name} consumes unmapped value "
                f"{value.name} (target={value.target!r}); typically a "
                "scalar kernel arg that wasn't resolvable for autodiff"
            )
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(args)
    visit(kwargs)


def _canonicalize_grad_outs(
    grad_outs: tuple[torch.Tensor, ...],
    target_ndims: tuple[int, ...],
    target_shapes: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reshape each grad_out to match the expected host-output shape.

    Needed when the forward reshapes outputs outside the tile loop (e.g.
    ``return out_2d.view(b, m, n)``); the generated bwd_fn was bound
    against the pre-reshape host-output shapes and must see the same
    ranks on every call.

    When ``target_shapes`` is provided, reshapes grad_out to the target
    shape when numels match and the shape differs (covers flatten/unflatten
    patterns). Falls back to squeezing size-1 dims for backward compat.
    """
    return tuple(
        _squeeze_to_target(
            g,
            n,
            i,
            target_shape=target_shapes[i] if target_shapes is not None else None,
        )
        for i, (g, n) in enumerate(zip(grad_outs, target_ndims, strict=True))
    )


def _squeeze_to_target(
    g: torch.Tensor,
    target_ndim: int,
    idx: int,
    target_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    # Fast path: already matches target shape.
    if target_shape is not None and tuple(g.shape) == target_shape:
        return g
    # If target_shape is provided and numels match, reshape directly.
    # This handles flatten/unflatten patterns like view(b, m, n) <-> view(b*m, n).
    if target_shape is not None and g.numel() == _numel(target_shape):
        return g.reshape(target_shape)
    # Expand scalar grad_out to target shape.  This handles the common
    # pattern where the kernel's wrapper applies ``torch.sum`` (whose
    # backward broadcasts the upstream scalar grad across the reduced
    # dimensions).
    if target_shape is not None and g.ndim < target_ndim:
        return g.expand(target_shape)
    # Fallback: squeeze size-1 dims to match target_ndim.
    while g.ndim > target_ndim:
        if g.shape[0] == 1:
            g = g.reshape(g.shape[1:])
        elif g.shape[-1] == 1:
            g = g.reshape(g.shape[:-1])
        else:
            raise exc.AutodiffNotSupported(
                f"grad_out {idx} shape {tuple(g.shape)} cannot reduce to "
                f"ndim={target_ndim} by squeezing size-1 dims"
            )
    return g


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _diff_squeezed_dims(
    in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Indices of the size-1 input dims that are absent from the output.

    Returns ``None`` when the input and output shapes can't be reconciled by
    pure size-1 removal (e.g. real shape changes or transposes).
    """
    squeezed: list[int] = []
    out_iter = iter(enumerate(out_shape))
    out_idx, out_size = next(out_iter, (None, None))
    for i, in_size in enumerate(in_shape):
        if out_idx is not None and out_size == in_size:
            out_idx, out_size = next(out_iter, (None, None))
            continue
        if in_size == 1:
            squeezed.append(i)
            continue
        return None
    if out_idx is not None:
        return None
    return tuple(squeezed)


_MODULE_NAMES: frozenset[str] = frozenset(
    {"torch", "helion", "hl", "math", "operator", "np", "numpy", "F"}
)


def _underlying_name(expr: ast.expr) -> str | None:
    """Walk ``.method(...)`` / ``.attr`` chains down to the leftmost Name.

    Returns None when the leftmost name is a known module namespace
    (e.g. torch, helion) rather than a tensor variable name — for example
    ``torch.stack(...)`` should not resolve to ``'torch'``.
    """
    while True:
        if isinstance(expr, ast.Name):
            name = expr.id
            if name in _MODULE_NAMES:
                return None  # module name, not a tensor
            return name
        if isinstance(expr, ast.Call):
            expr = expr.func
        elif isinstance(expr, ast.Attribute):
            expr = expr.value
        else:
            return None


def _extract_derivation_map(
    host_function: object | None,
) -> dict[str, tuple[str, list[tuple[str, list]]]]:
    """Walk the kernel's host AST to find ``derived = source.method(...)``
    chains. Returns ``{derived_name: (source_name, [(method, args), ...])}``
    where the list captures the chain of method calls (in forward order)."""
    import ast as _ast

    result: dict[str, tuple[str, list[tuple[str, list]]]] = {}
    if host_function is None:
        return result
    body = getattr(host_function, "body", None)
    if not body:
        return result
    for stmt in body:
        if not isinstance(stmt, _ast.Assign) or len(stmt.targets) != 1:
            continue
        lhs = stmt.targets[0]
        if not isinstance(lhs, _ast.Name):
            continue
        node = stmt.value
        chain: list[tuple[str, list]] = []
        while isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute):
            method = node.func.attr
            args: list = []
            for a in node.args:
                if isinstance(a, _ast.Constant):
                    args.append(a.value)
                elif (
                    isinstance(a, _ast.UnaryOp)
                    and isinstance(a.op, _ast.USub)
                    and isinstance(a.operand, _ast.Constant)
                ):
                    # Negative integer literals (e.g. -1, -2) are represented
                    # as UnaryOp(USub, Constant(n)), not Constant(-n).
                    args.append(-(a.operand.value))  # pyrefly: ignore [unsupported-operation]
                elif isinstance(a, _ast.List):
                    args.append(
                        [
                            e.value
                            if isinstance(e, _ast.Constant)
                            else -(e.operand.value)  # pyrefly: ignore [unsupported-operation]
                            if isinstance(e, _ast.UnaryOp)
                            and isinstance(e.op, _ast.USub)
                            and isinstance(e.operand, _ast.Constant)
                            else None
                            for e in a.elts
                        ]
                    )
                elif isinstance(a, _ast.Tuple):
                    args.append(
                        tuple(
                            e.value if isinstance(e, _ast.Constant) else None
                            for e in a.elts
                        )
                    )
            # Handle keyword arguments for known methods.
            # e.g. x.view(size=(4, -1)) or x.transpose(dim0=0, dim1=1).
            _KEYWORD_ORDER: dict[str, list[str]] = {
                "transpose": ["dim0", "dim1"],
                "permute": ["dims"],
                "view": ["size"],
                "reshape": ["shape"],
                "squeeze": ["dim"],
                "unsqueeze": ["dim"],
            }

            def _parse_kw_scalar(kval: _ast.expr) -> int | None:
                if isinstance(kval, _ast.Constant) and isinstance(kval.value, int):
                    return kval.value
                if (
                    isinstance(kval, _ast.UnaryOp)
                    and isinstance(kval.op, _ast.USub)
                    and isinstance(kval.operand, _ast.Constant)
                    and isinstance(kval.operand.value, int)
                ):
                    return -kval.operand.value
                return None

            def _parse_kw_seq(kval: _ast.expr) -> list | None:
                if not isinstance(kval, (_ast.List, _ast.Tuple)):
                    return None
                result_seq = []
                for e in kval.elts:
                    v = _parse_kw_scalar(e)
                    result_seq.append(v)
                return result_seq

            if node.keywords and method in _KEYWORD_ORDER:
                kw_map = {
                    kw.arg: kw.value for kw in node.keywords if kw.arg is not None
                }
                for kname in _KEYWORD_ORDER[method]:
                    kval = kw_map.get(kname)
                    if kval is None:
                        continue
                    scalar = _parse_kw_scalar(kval)
                    if scalar is not None:
                        args.append(scalar)
                    else:
                        seq = _parse_kw_seq(kval)
                        if seq is not None:
                            args.extend(seq)
            chain.append((method, args))
            node = node.func.value
        if isinstance(node, _ast.Name) and node.id not in _MODULE_NAMES:
            chain.reverse()
            source_name = node.id
            # Guard against self-referential derivations: ``out = out.view(-1)``
            # produces source_name == lhs.id, which creates an unresolvable
            # cycle.  Transitively resolve through the prior entry when one
            # exists; otherwise drop this entry.
            if source_name == lhs.id:
                prior = result.get(lhs.id)
                if prior is not None:
                    # Compose: prior chain then this chain.
                    result[lhs.id] = (prior[0], prior[1] + chain)
                # else: no prior info, skip (can't resolve)
            else:
                result[lhs.id] = (source_name, chain)
    return result


def _apply_inverse_derivation(
    grad: torch.Tensor,
    chain: list[tuple[str, list]],
    original_shape: tuple[int, ...],
) -> torch.Tensor:
    """Apply the inverse of a derivation chain to a gradient tensor.

    Forward-simulates the chain on a meta tensor to compute the shape
    BEFORE each forward step, then applies the inverse of each operation
    in reverse order using those intermediate shapes as reshape targets.
    Handles transpose, permute, t, view/reshape/flatten,
    squeeze/unsqueeze, and contiguous. Unhandled methods raise
    ``AutodiffNotSupported``.
    """

    def _normalize_flat(args: list) -> list:
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return list(args[0])
        return list(args)

    def _replace_none(args: list, fallback_shape: tuple[int, ...] | None) -> list:
        out = []
        for a in args:
            if a is None:
                out.append(-1)
            else:
                out.append(a)
        # Multiple -1 are not legal in reshape. Use fallback to fill.
        if out.count(-1) > 1 and fallback_shape is not None:
            i = 0
            for k, v in enumerate(out):
                if v == -1:
                    out[k] = fallback_shape[i]
                    i += 1
                else:
                    i += 1
        return out

    # Pre-compute the expected grad shape at each forward step by working
    # BACKWARDS from grad.shape through the inverse of each op.  This gives us
    # a fallback for reshapes whose args contain None (dynamic variable names
    # that _extract_derivation_map could not resolve to concrete ints, e.g.
    # ``q_in.reshape([-1, m_dim, head_dim])`` → args = [[-1, None, None]]).
    # For such reshapes, the forward simulation cannot infer the result shape,
    # but we know the result equals the expected intermediate shape derived here.
    _expected_after: list[tuple[int, ...]] = []
    _back_shape = tuple(grad.shape)
    for _method, _args in reversed(chain):
        _expected_after.insert(0, _back_shape)
        _flat = _normalize_flat(_args)
        if _method == "transpose":
            if len(_flat) >= 2:
                _back_shape = tuple(
                    torch.empty(_back_shape, device="meta")
                    .transpose(_flat[0], _flat[1])
                    .shape
                )
        elif _method == "permute":
            _inv = [0] * len(_flat)
            for _j, _p in enumerate(_flat):
                _inv[_p] = _j
            _back_shape = tuple(
                torch.empty(_back_shape, device="meta").permute(_inv).shape
            )
        elif _method == "t":
            _back_shape = tuple(torch.empty(_back_shape, device="meta").t().shape)
        elif _method == "unsqueeze":
            if _flat:
                _back_shape = tuple(
                    torch.empty(_back_shape, device="meta").squeeze(_flat[0]).shape
                )
        elif _method == "squeeze":
            # Inverse of squeeze is unsqueeze; for the backward pass we want
            # the shape BEFORE the squeeze (= with the dim restored).
            pass  # Can't easily infer unsqueeze position; leave _back_shape as-is
        elif _method in ("view", "reshape", "flatten"):
            # Can't invert a reshape without knowing the pre-step shape; leave as-is.
            pass
        # For "contiguous" and other no-ops, _back_shape is unchanged.
    # _back_shape is now the expected shape BEFORE the first chain step = original_shape.
    # (Approximate for reshape/flatten; the final grad.reshape(original_shape) handles it.)

    # Forward simulate to record shape before each step.
    sim = torch.empty(original_shape, device="meta")
    shapes_fwd: list[tuple[int, ...]] = []
    for _step_i, (method, args) in enumerate(chain):
        shapes_fwd.append(tuple(sim.shape))
        flat = _normalize_flat(args)
        try:
            if method in ("view", "reshape"):
                target = _replace_none(flat, None)
                sim = sim.reshape(target)
            elif method == "flatten":
                sim = sim.flatten(*flat) if flat else sim.flatten()
            elif method == "transpose":
                if len(flat) >= 2:
                    sim = sim.transpose(flat[0], flat[1])
            elif method == "permute":
                sim = sim.permute(flat)
            elif method == "t":
                sim = sim.t()
            elif method == "squeeze":
                sim = sim.squeeze(*flat) if flat else sim.squeeze()
            elif method == "unsqueeze":
                if flat:
                    sim = sim.unsqueeze(flat[0])
            elif method == "contiguous":
                pass
            else:
                raise exc.AutodiffNotSupported(
                    f"_apply_inverse_derivation: unhandled method '{method}'"
                )
        except (RuntimeError, IndexError, TypeError) as e:
            # reshape args may contain None for dynamic variable names
            # (e.g. ``q_in.reshape([-1, m_dim, head_dim])`` → args=[[-1, None, None]]);
            # the simulation cannot infer the result shape, so use the
            # expected-after shape derived from working backwards through grad.shape.
            if method in ("view", "reshape", "flatten"):
                expected = _expected_after[_step_i]
                sim = torch.empty(expected, device="meta")
            else:
                raise exc.AutodiffNotSupported(
                    f"_apply_inverse_derivation: failed to simulate '{method}' with args {args}: {e}"
                ) from e

    # Apply inverse in reverse order.
    for i in range(len(chain) - 1, -1, -1):
        method, args = chain[i]
        flat = _normalize_flat(args)
        target_shape = shapes_fwd[i]
        if method in ("view", "reshape", "flatten"):
            grad = grad.reshape(target_shape)
        elif method == "transpose":
            if len(flat) >= 2:
                grad = grad.transpose(flat[0], flat[1])
        elif method == "permute":
            inv = [0] * len(flat)
            for j, p in enumerate(flat):
                inv[p] = j
            grad = grad.permute(inv)
        elif method == "t":
            grad = grad.t()
        elif method == "squeeze":
            # Forward dropped a dim of size 1; inverse: unsqueeze at that pos.
            if flat:
                grad = grad.unsqueeze(flat[0])
            else:
                # Argless squeeze drops all size-1 dims. Reshape to target_shape
                # (which has the size-1 dims restored).
                grad = grad.reshape(target_shape)
        elif method == "unsqueeze":
            if flat:
                grad = grad.squeeze(flat[0])
        elif method == "contiguous":
            pass
        else:
            raise exc.AutodiffNotSupported(
                f"_apply_inverse_derivation: unhandled method '{method}'"
            )
    return grad


def _reorder_inputs(
    tensor_inputs: tuple[torch.Tensor, ...],
    kernel_names: list[str],
    input_mappings: list,
    host_function: object | None,
) -> tuple[torch.Tensor, ...]:
    """Reorder tensor_inputs to match input_mappings order, resolving
    host-derived tensors via AST derivation tracking."""
    if not input_mappings:
        return tensor_inputs
    derivation_map = _extract_derivation_map(host_function)
    host_tensors: dict[str, torch.Tensor] = {}
    if host_function is not None:
        for fake, origin in getattr(host_function, "tensor_to_origin", {}).items():
            name = getattr(origin, "name", None)
            if name and isinstance(fake, torch.Tensor):
                host_tensors[name] = fake
    result: list[torch.Tensor] = []
    for m in input_mappings:
        if m.tensor_name in kernel_names:
            result.append(tensor_inputs[kernel_names.index(m.tensor_name)])
        else:
            deriv = derivation_map.get(m.tensor_name)
            fake = host_tensors.get(m.tensor_name)
            source = deriv[0] if deriv else None
            chain = deriv[1] if deriv else []
            # Transitively resolve multi-level derivations: if the direct
            # source is not a kernel input but has its own derivation entry,
            # compose the chains and resolve recursively.
            while source and source not in kernel_names:
                parent_deriv = derivation_map.get(source)
                if parent_deriv is None:
                    break
                # Compose: parent chain then current chain.
                source = parent_deriv[0]
                chain = parent_deriv[1] + chain
            if source and source in kernel_names and fake is not None:
                src_tensor = tensor_inputs[kernel_names.index(source)]
                final_shape = tuple(int(s) for s in fake.shape)

                # For reshape/view ops with fully concrete args (no None/dynamic
                # dims), use the args directly as the target shape: inverting
                # from final_shape is wrong when a second reshape appears later
                # in the chain (it only un-applies transpositions, skipping
                # reshapes).  For dynamic args (containing None), fall back to
                # the invert-from-final_shape approach which correctly handles
                # the transpose-only suffix case.
                def _norm_flat(a: list) -> list:
                    """Normalise ``(method, args)`` args to a flat list.
                    args=[[-1,N,M]] (list-in-list) → [-1,N,M]; [3,4] → [3,4].
                    """
                    if len(a) == 1 and isinstance(a[0], (list, tuple)):
                        return list(a[0])
                    return list(a)

                t = src_tensor
                for ci, (method, args) in enumerate(chain):
                    if method in ("reshape", "view"):
                        flat = _norm_flat(
                            list(args) if isinstance(args, (list, tuple)) else [args]
                        )
                        has_dynamic = any(a is None for a in flat)
                        if not has_dynamic and flat:
                            # Concrete args → use them directly.
                            t = t.reshape(flat)
                        else:
                            # Dynamic reshape: compute target by inverting
                            # the suffix ops (transpose-only) from final_shape.
                            intermediate = list(final_shape)
                            for rm, ra in reversed(chain[ci + 1 :]):
                                if rm == "transpose" and len(ra) >= 2:
                                    intermediate[ra[0]], intermediate[ra[1]] = (
                                        intermediate[ra[1]],
                                        intermediate[ra[0]],
                                    )
                            t = t.reshape(intermediate)
                    elif method == "transpose" and len(args) >= 2:
                        t = t.transpose(args[0], args[1])
                    elif method == "contiguous":
                        t = t.contiguous()
                    else:
                        t = getattr(t, method)(*args)
                result.append(t)
            elif fake is not None:
                # Fallback: numel match.
                target_shape = tuple(int(s) for s in fake.shape)
                target_numel = 1
                for s in target_shape:
                    target_numel *= s
                matched = False
                for ki in range(len(kernel_names)):
                    if (
                        tensor_inputs[ki].numel() == target_numel
                        and tensor_inputs[ki].dtype == fake.dtype
                    ):
                        result.append(tensor_inputs[ki].reshape(target_shape))
                        matched = True
                        break
                if not matched:
                    result.append(tensor_inputs[0].new_empty(target_shape))
            else:
                result.append(tensor_inputs[0].new_empty(0))
    return tuple(result)


def _make_init_tensors(
    init_specs: list[InitSpec],
    tensor_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    out: list[torch.Tensor] = []
    for spec in init_specs:
        shape: list[int] = []
        for src in spec.shape_dim_sources:
            if isinstance(src, tuple):
                idx, dim = src
                shape.append(int(tensor_inputs[idx].shape[dim]))
            else:
                shape.append(int(src))
        rg = spec.dtype.is_floating_point or spec.dtype.is_complex
        out.append(
            torch.full(
                shape,
                spec.fill_value,
                dtype=spec.dtype,
                device=spec.device,
                requires_grad=rg,
            )
        )
    return tuple(out)


def backward(
    kernel: Kernel[object],
    grad_out: torch.Tensor | tuple[torch.Tensor, ...],
    *inputs: object,
    return_code: bool = False,
    autotune: bool = False,
    autotune_effort: str | None = None,
) -> (
    tuple[torch.Tensor, ...]
    | torch.Tensor
    | tuple[tuple[torch.Tensor, ...] | torch.Tensor, str, str]
):
    """
    Compute gradients for a Helion kernel.

    The backward kernel is generated as an independent Helion kernel with its
    own ConfigSpec, allowing separate autotuning from the forward kernel.

    Supports a single tile loop with elementwise ops and the reductions
    ``sum``, ``mean``, ``amax``, ``amin``. Other reductions (``prod``,
    ``var``, ``std``, ``argmax``/``argmin``, ``cumsum``, ``cumprod``, ...)
    raise :class:`~helion.exc.AutodiffNotSupported`.

    Args:
        kernel: A @helion.kernel decorated function (must be called once first)
        grad_out: Gradient of loss w.r.t. kernel output. For multi-output kernels,
            pass a tuple of gradient tensors (one per output).
        *inputs: The original inputs to the kernel in the same order as forward.
            Pass scalar arguments (e.g. ``eps``) here too — they are read for
            symnode resolution and stripped before the backward kernel runs.
        return_code: If True, also return the generated backward kernel code
        autotune: If True, autotune the backward kernel for best performance
        autotune_effort: Autotuning effort level ('none', 'quick', 'full').
            Default is 'none' when autotune=False, 'quick' when autotune=True.

    Returns:
        If return_code=False: Tuple of gradients (or single tensor if one input)
        If return_code=True: (gradients, helion_code, triton_code) tuple

    Example:
        @helion.kernel()
        def my_kernel(x, y):
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * y[tile]
            return out

        out = my_kernel(x, y)
        grad_x, grad_y = helion.experimental.backward(my_kernel, grad_out, x, y)
    """
    # Pre-validate for var/std BEFORE checking AutodiffKernelNotCalled.
    # ``var``/``std`` decompose to ``_inductor_lowering_extra`` nodes that cause
    # an opaque ``InductorLoweringError`` at forward-kernel codegen time; the
    # user sees this from the *forward* call, never reaching backward().  By
    # running ``_validate_supported_reductions`` here via a fresh bind() we
    # surface a clear ``AutodiffNotSupported`` error from backward() regardless
    # of whether the forward compiled successfully.
    try:
        _pre_bound = kernel.bind(inputs)
        if _pre_bound.host_function is not None:
            from .._compiler.device_ir import RootGraphInfo as _RGI_pre

            for _gi_pre in _pre_bound.host_function.device_ir.graphs:
                if isinstance(_gi_pre, _RGI_pre):
                    _validate_supported_reductions(_gi_pre.graph)
                    break
    except exc.AutodiffNotSupported:
        raise
    except Exception:
        pass  # ignore graph-inspection errors; real errors will surface below

    if not hasattr(kernel, "_bound_kernels") or not kernel._bound_kernels:
        raise exc.AutodiffKernelNotCalled

    if isinstance(grad_out, torch.Tensor):
        grad_outs = (grad_out,)
    else:
        grad_outs = tuple(grad_out)

    # Split tensor inputs (used by the generated backward kernel) from
    # scalar inputs (used only to resolve `_get_symnode` references). The
    # forward bind call needs all of them, in original order.
    tensor_inputs = tuple(t for t in inputs if isinstance(t, torch.Tensor))
    bound = kernel.bind(inputs)
    if bound._config is None:
        bound._config = bound.env.config_spec.default_config()

    if autotune_effort is None:
        autotune_effort = "quick" if autotune else "none"

    # Include scalar args in cache key so different eps/beta/etc. values
    # don't silently reuse a backward kernel with baked-in constants. Make each
    # value hashable: non-tensor args can be lists/dicts (e.g. layer_norm's
    # `normalized_shape=[N]`), which would otherwise crash the cache-key hash.
    def _hashable(v: object) -> object:
        if isinstance(v, (list, tuple)):
            return tuple(_hashable(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
        try:
            hash(v)
        except TypeError:
            return repr(v)
        return v

    scalar_args = tuple(_hashable(v) for v in inputs if not isinstance(v, torch.Tensor))

    # Include forward block sizes in cache key: for scan-hop (multi-tile-loop)
    # kernels the backward source has forward block sizes BAKED IN (e.g.
    # reshape shapes), so a different block-size config on the same `bound`
    # object would silently reuse a stale backward kernel.
    _fwd_block_sizes: tuple[tuple[int, int], ...] = ()
    if bound._config is not None and bound._config.block_sizes is not None:
        with contextlib.suppress(Exception):
            _fwd_block_sizes = tuple(sorted(enumerate(bound._config.block_sizes)))

    # Include device index in cache key: the forward BoundKernel specialisation
    # key uses only device.type (e.g. "cuda"), so cuda:0 and cuda:1 share the
    # same BoundKernel and backward cache.  The backward source has device
    # literals baked in from whichever device ran first, so we must key on the
    # full device spec of the tensor inputs.
    _device_key: tuple[tuple[str, int | None], ...] = tuple(
        (t.device.type, t.device.index) for t in tensor_inputs
    )

    cache_key = (autotune, autotune_effort, scalar_args, _fwd_block_sizes, _device_key)
    cache: dict[tuple, tuple[Any, ...]] | None = getattr(
        bound, "_backward_compiled_cache", None
    )
    kernel_tensor_input_names: list[str] = []
    cached_init_specs: list = []
    cached_input_mappings: list = []
    cached_kernel_names: list[str] = []
    cached_host_function: object | None = None
    target_shapes: tuple[tuple[int, ...], ...] | None = None
    # Pre-declare backward-kernel locals so the nested _unpack_cached function
    # can reference them via `nonlocal` (Python requires the binding to exist
    # in the enclosing scope at function-definition time).
    bwd_fn: object = None
    bwd_source: str = ""
    bwd_bound: object = None
    target_ndims: tuple[int, ...] = ()
    is_scan_hop: bool = False

    def _unpack_cached(cached: tuple) -> None:
        """Unpack a cache entry into the surrounding locals (via nonlocal).

        The cache is per-process / per-bound-kernel and is only ever written as
        the 10-tuple below (single write site), so unpack it directly.
        """
        nonlocal \
            bwd_fn, \
            bwd_source, \
            bwd_bound, \
            target_ndims, \
            is_scan_hop, \
            cached_init_specs, \
            cached_input_mappings, \
            cached_kernel_names, \
            cached_host_function, \
            target_shapes
        (
            bwd_fn,
            bwd_source,
            bwd_bound,
            target_ndims,
            is_scan_hop,
            cached_init_specs,
            cached_input_mappings,
            cached_kernel_names,
            cached_host_function,
            target_shapes,
        ) = cached

    # Fast-path cache read (unlocked) avoids lock contention on the hot path.
    # The slow path (compilation) is serialised by the lock so that (a)
    # concurrent threads don't race through AOT-autograd (which has non-thread-
    # safe internal state) and (b) the TOCTOU race when initialising
    # bound._backward_compiled_cache is prevented.
    _cache_hit = cache is not None and cache_key in cache
    if _cache_hit:
        _unpack_cached(cache[cache_key])
    else:
        # Acquire lock, then double-check: another thread may have compiled
        # while we were waiting.
        with _backward_cache_lock:
            cache = getattr(bound, "_backward_compiled_cache", None)
            if cache is not None and cache_key in cache:
                _unpack_cached(cache[cache_key])
                _cache_hit = True

    if not _cache_hit:
        from .._compiler.device_ir import RootGraphInfo

        host_function = bound.host_function
        assert host_function is not None
        graphs = host_function.device_ir.graphs

        root_graph_info = None
        for graph_info in graphs:
            if isinstance(graph_info, RootGraphInfo):
                if root_graph_info is not None:
                    raise exc.AutodiffNotSupported("multiple root graphs")
                root_graph_info = graph_info
        if root_graph_info is None:
            raise exc.AutodiffNotSupported("no root graph found")

        fwd_graph = root_graph_info.graph
        _validate_supported_reductions(fwd_graph)
        _validate_no_subscript_permutation(list(graphs))

        scalar_values = _resolve_scalar_values(kernel, inputs, host_function)
        return_order = _extract_return_order(host_function)
        if return_order is None and len(grad_outs) > 1:
            # Without return-order info we can't pair multi-output grads
            # correctly — store-order fallback would silently mis-pair.
            raise exc.AutodiffNotSupported(
                "multi-output kernel with a return value too complex to "
                "match host-tensor names; rewrite the return as a tuple "
                "of names or simple `name.method(...)` chains"
            )
        forward_reduce_dims = _extract_forward_reduce_dims(fwd_graph)
        forward_full_slice_dims = _extract_forward_full_slice_dims(fwd_graph)
        forward_untiled_inputs = _extract_forward_untiled_inputs(fwd_graph)
        has_root_matmul = _forward_has_root_matmul(fwd_graph)

        kernel_tensor_input_names = [
            nm
            for nm, t in zip(
                inspect.signature(kernel.fn).parameters.keys(),
                inputs,
                strict=False,
            )
            if isinstance(t, torch.Tensor)
        ]
        # Build config_block_sizes from both auto-tuned block_ids and
        # fixed/static block sizes (e.g. user-specified chunk_size).
        config_block_sizes: dict[int, int] = {
            bid: bound._config.block_sizes[
                bound.env.config_spec.block_sizes.block_id_to_index(bid)
            ]
            for bid in bound.env.config_spec.block_sizes.valid_block_ids()
        }
        from .._compiler.compile_environment import FixedBlockSizeSource

        for bs_info in bound.env.block_sizes:
            if bs_info.block_id not in config_block_sizes and isinstance(
                bs_info.block_size_source, FixedBlockSizeSource
            ):
                config_block_sizes[bs_info.block_id] = int(
                    bs_info.block_size_source.value
                )
        analyzer = GraphAnalyzer(
            fwd_graph,
            scalar_values=scalar_values,
            return_order=return_order,
            all_graphs=list(graphs),
            config_block_sizes=config_block_sizes,
            kernel_input_names=kernel_tensor_input_names,
        )
        compute_graph, input_mappings, compute_output_shapes, host_output_shapes = (
            analyzer.extract_computation_graph()
        )

        target_ndims = tuple(len(s) for s in compute_output_shapes)
        # Use host output shapes for canonicalization: the user's grad_out
        # matches the kernel's return shape (host level), not the per-tile
        # compute graph shape.
        target_shapes = tuple(host_output_shapes)
        # Canonicalize grad_outs exactly once on the common path (after the
        # if/else block) rather than redundantly inside the cache-miss block.
        # Validate count mismatch here early to emit a helpful error before
        # attempting AOT differentiation.
        # When the user provides fewer grad_outs than the kernel has outputs
        # (e.g. backward(se_block_fwd, grad_out_for_out, x, w) where
        # se_block_fwd returns (out, s)), pad missing grad_outs with zeros so
        # AOT autograd sees a complete gradient signal.
        if len(grad_outs) < len(target_ndims):
            _ref_t = grad_outs[0]
            extra = tuple(
                torch.zeros(
                    host_output_shapes[i],
                    dtype=_ref_t.dtype,
                    device=_ref_t.device,
                )
                for i in range(len(grad_outs), len(target_ndims))
            )
            grad_outs = grad_outs + extra
        if len(grad_outs) != len(target_ndims):
            raise exc.AutodiffNotSupported(
                f"grad_out / compute_graph output count mismatch "
                f"({len(grad_outs)} grad_outs vs {len(target_ndims)} outputs)"
            )

        # ScanAutogradOp zeros the carry-grad chain unless inits carry
        # `requires_grad=True`; build leaf init tensors here.
        init_tensors_by_name: dict[str, torch.Tensor] = {}
        for spec in analyzer.init_specs:
            shape = []
            for src in spec.shape_dim_sources:
                if isinstance(src, tuple):
                    inp_idx, dim = src
                    shape.append(int(tensor_inputs[inp_idx].shape[dim]))
                else:
                    shape.append(int(src))
            init_tensors_by_name[spec.placeholder_name] = torch.full(
                shape,
                spec.fill_value,
                dtype=spec.dtype,
                device=spec.device,
                requires_grad=True,
            )

        input_mapping_by_name = {m.placeholder_name: m for m in input_mappings}
        derivation_map = _extract_derivation_map(host_function)
        derived_tensor_cache: dict[str, torch.Tensor] = {}
        for m in input_mappings:
            if m.tensor_name in kernel_tensor_input_names:
                continue
            deriv = derivation_map.get(m.tensor_name)
            source = deriv[0] if deriv else None
            chain = deriv[1] if deriv else []
            if source and source in kernel_tensor_input_names:
                src_tensor = tensor_inputs[kernel_tensor_input_names.index(source)]
                host_tensors_local: dict[str, torch.Tensor] = {}
                for fake_t, origin in host_function.tensor_to_origin.items():
                    nm = getattr(origin, "name", None)
                    if nm and isinstance(fake_t, torch.Tensor):
                        host_tensors_local[nm] = fake_t
                fake = host_tensors_local.get(m.tensor_name)
                final_shape = (
                    tuple(int(s) for s in fake.shape) if fake is not None else ()
                )
                t = src_tensor
                for ci, (method, args) in enumerate(chain):
                    if method in ("reshape", "view"):
                        intermediate = list(final_shape)
                        for rm, ra in reversed(chain[ci + 1 :]):
                            if rm == "transpose" and len(ra) >= 2:
                                intermediate[ra[0]], intermediate[ra[1]] = (
                                    intermediate[ra[1]],
                                    intermediate[ra[0]],
                                )
                        t = t.reshape(intermediate)
                    elif method == "transpose" and len(args) >= 2:
                        t = t.transpose(args[0], args[1])
                    elif method == "contiguous":
                        t = t.contiguous()
                    else:
                        t = getattr(t, method)(*args)
                derived_tensor_cache[m.tensor_name] = t

        ordered_inputs: list[torch.Tensor] = []
        for ph in compute_graph.find_nodes(op="placeholder"):
            if ph.name in init_tensors_by_name:
                ordered_inputs.append(init_tensors_by_name[ph.name])
            elif ph.name in input_mapping_by_name:
                tname = input_mapping_by_name[ph.name].tensor_name
                if tname in kernel_tensor_input_names:
                    ordered_inputs.append(
                        tensor_inputs[kernel_tensor_input_names.index(tname)]
                    )
                elif tname in derived_tensor_cache:
                    ordered_inputs.append(derived_tensor_cache[tname])
                else:
                    raise exc.AutodiffNotSupported(
                        f"host-derived tensor {tname!r} cannot be resolved "
                        "to a kernel input"
                    )
            else:
                raise exc.AutodiffNotSupported(
                    f"compute_graph placeholder {ph.name!r} unmapped"
                )

        combine_attrs = {
            f"scan_combine_graph_{i}": gm for i, gm in enumerate(analyzer.scan_combines)
        }

        bw_module_holder: list = []
        try:
            backward_graph = differentiate_graph(
                compute_graph,
                tuple(ordered_inputs),
                combine_attrs=combine_attrs,
                bw_module_holder=bw_module_holder,
            )
        except exc.AutodiffNotSupported:
            raise
        except (
            RuntimeError,
            AssertionError,
            ValueError,
            AttributeError,
            TypeError,
        ) as e:
            raise exc.AutodiffNotSupported(
                f"scan-based autodiff failed during AOT differentiation: {e}"
            ) from e
        bw_module = bw_module_holder[0] if bw_module_holder else None

        # Canonicalize grad_outs before the converter sees their shapes so that
        # FXToHelionConverter receives compute-graph shapes, not host-level
        # post-reshape shapes.  When the forward kernel returns a reshaped view
        # (e.g. `inv_rms.reshape(-1, 1)`), the user supplies a grad_out
        # matching the HOST return shape, while the compute graph was
        # differentiated against the PER-TILE shape.  Using the wrong ndim
        # produces bad subscript expressions like `grad_out_1[tile_0, :]` for
        # a 1-D param, causing shape errors.
        #
        # The common-path canonicalization below is kept for cache hits
        # (it is a no-op here since target_shapes and target_ndims are
        # identical after this point).
        try:
            grad_outs = _canonicalize_grad_outs(
                grad_outs, target_ndims, target_shapes=target_shapes
            )
        except (ValueError, RuntimeError) as e:
            raise exc.AutodiffNotSupported(
                f"grad_out / compute_graph output shape mismatch "
                f"({[g.shape for g in grad_outs]} vs {target_shapes}): {e}"
            ) from e

        # Build input_tensors in input_mappings order (may include
        # host-derived tensors like q_view).
        converter_inputs = tuple(
            tensor_inputs[kernel_tensor_input_names.index(m.tensor_name)]
            if m.tensor_name in kernel_tensor_input_names
            else derived_tensor_cache[m.tensor_name]
            for m in input_mappings
        )
        converter = FXToHelionConverter(
            backward_graph=backward_graph,
            input_mappings=input_mappings,
            input_tensors=converter_inputs,
            grad_out_shapes=tuple(g.shape for g in grad_outs),
            forward_reduce_dims=forward_reduce_dims,
            forward_full_slice_dims=forward_full_slice_dims,
            forward_untiled_inputs=forward_untiled_inputs,
            has_root_matmul=has_root_matmul,
            bw_module=bw_module,
            init_specs=analyzer.init_specs,
            compute_graph=compute_graph,
        )
        try:
            bwd_source = converter.convert()
        except exc.AutodiffNotSupported:
            raise
        except (RuntimeError, AssertionError, ValueError) as e:
            raise exc.AutodiffNotSupported(
                f"backward kernel generation failed: {e}"
            ) from e

        # Stable on-disk path so `inspect.getsource` and tracebacks still
        # work after we return; write-temp-then-replace for safe concurrent
        # writers.
        from ..autotuner.local_cache import get_helion_cache_dir

        # Use the full 32-char MD5 hexdigest to reduce birthday-paradox
        # collision probability (48-bit truncation gives ~1 collision after
        # sqrt(2^48) ≈ 16M distinct sources).
        source_hash = hashlib.md5(
            bwd_source.encode(), usedforsecurity=False
        ).hexdigest()  # full 32 hex chars
        cache_dir = get_helion_cache_dir() / "backward"
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_path = cache_dir / f"helion_bwd_{source_hash}.py"
        if not source_path.exists():
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".helion_bwd_{source_hash}.",
                suffix=".py.tmp",
                dir=str(cache_dir),
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(bwd_source)
                os.replace(tmp_path, source_path)
            except BaseException:
                pathlib.Path(tmp_path).unlink(missing_ok=True)
                raise
        else:
            # Verify the on-disk content matches the expected source to catch
            # hash collisions or file corruption before loading.
            try:
                disk_content = source_path.read_text()
            except OSError:
                disk_content = ""
            if disk_content != bwd_source:
                # Content mismatch: overwrite with the correct source so
                # subsequent calls load the right kernel.
                fd, tmp_path = tempfile.mkstemp(
                    prefix=f".helion_bwd_{source_hash}.",
                    suffix=".py.tmp",
                    dir=str(cache_dir),
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        f.write(bwd_source)
                    os.replace(tmp_path, source_path)
                except BaseException:
                    pathlib.Path(tmp_path).unlink(missing_ok=True)
                    raise

        spec = importlib.util.spec_from_file_location(
            f"helion_bwd_{source_hash}", str(source_path)
        )
        assert spec is not None and spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        # If exec_module fails (e.g. SyntaxError or module-level exception in
        # generated source), delete the cached file so it gets regenerated on
        # the next call instead of poisoning the disk cache.
        try:
            spec.loader.exec_module(module)
        except BaseException:
            source_path.unlink(missing_ok=True)
            raise

        assert hasattr(module, "backward_kernel")

        bwd_fn = module.backward_kernel
        is_scan_hop = bool(analyzer.scan_combines)
        cached_init_specs = list(analyzer.init_specs)

        if is_scan_hop:
            bwd_args = (
                *grad_outs,
                *converter_inputs,
                *_make_init_tensors(cached_init_specs, tensor_inputs),
                *(torch.zeros_like(t) for t in converter_inputs),
            )
        else:
            bwd_args = (*grad_outs, *converter_inputs)
        bwd_bound = bwd_fn.bind(bwd_args)

        bwd_fn.settings.autotune_effort = autotune_effort  # pyrefly: ignore [missing-attribute]
        if autotune:
            bwd_fn.autotune(bwd_args)

        # Serialise cache initialisation with the module-level lock to
        # eliminate the TOCTOU race between (A) reading cache=None,
        # (B) creating a new dict and (C) assigning it to bound.
        with _backward_cache_lock:
            cache = getattr(bound, "_backward_compiled_cache", None)
            if cache is None:
                cache = {}
                bound._backward_compiled_cache = cache  # pyrefly: ignore [missing-attribute]
        cached_input_mappings = input_mappings
        cached_kernel_names = kernel_tensor_input_names
        cached_host_function = host_function
        cache[cache_key] = (
            bwd_fn,
            bwd_source,
            bwd_bound,
            target_ndims,
            is_scan_hop,
            cached_init_specs,
            cached_input_mappings,
            cached_kernel_names,
            cached_host_function,
            target_shapes,
        )

    # Pad missing grad_outs with zeros on the COMMON path so a multi-output
    # kernel called with a single grad_out also works on cache HITS (the
    # cache-miss block above pads only for AOT differentiation; cache hits skip
    # it). Without this, _canonicalize_grad_outs' strict zip raised on the 2nd
    # and later calls.
    if target_shapes is not None and len(grad_outs) < len(target_shapes):
        _ref = grad_outs[0]
        grad_outs = grad_outs + tuple(
            torch.zeros(target_shapes[i], dtype=_ref.dtype, device=_ref.device)
            for i in range(len(grad_outs), len(target_shapes))
        )
    grad_outs = _canonicalize_grad_outs(
        grad_outs, target_ndims, target_shapes=target_shapes
    )
    knames = cached_kernel_names or kernel_tensor_input_names
    hf = cached_host_function
    rt_inputs = _reorder_inputs(
        tensor_inputs,
        knames,
        cached_input_mappings,
        hf,
    )
    if is_scan_hop:
        grad_buffers = tuple(torch.zeros_like(t) for t in rt_inputs)
        bwd_fn(  # pyrefly: ignore [not-callable]
            *grad_outs,
            *rt_inputs,
            *_make_init_tensors(cached_init_specs, tensor_inputs),
            *grad_buffers,
        )
        derivation_map = _extract_derivation_map(hf)
        final_grads: list[torch.Tensor] = [torch.zeros_like(t) for t in tensor_inputs]
        for gi, m in enumerate(cached_input_mappings):
            if m.tensor_name in knames:
                ki = knames.index(m.tensor_name)
                final_grads[ki] = final_grads[ki] + grad_buffers[gi].reshape(
                    tensor_inputs[ki].shape
                )
            else:
                deriv = derivation_map.get(m.tensor_name)
                source = deriv[0] if deriv else None
                chain = deriv[1] if deriv else []
                if source and source in knames:
                    ki = knames.index(source)
                    final_grads[ki] = final_grads[ki] + _apply_inverse_derivation(
                        grad_buffers[gi], chain, tensor_inputs[ki].shape
                    )
        grads: torch.Tensor | tuple[torch.Tensor, ...] = (
            tuple(final_grads) if len(final_grads) > 1 else final_grads[0]
        )
    else:
        result = bwd_fn(*grad_outs, *rt_inputs)  # pyrefly: ignore [not-callable]
        if isinstance(result, tuple):
            assert all(isinstance(r, torch.Tensor) for r in result)
            result_list = list(result)
        else:
            assert isinstance(result, torch.Tensor)
            result_list = [result]
        # Reshape grad results back to original input shapes. The compute
        # graph may work with host-derived tensors (e.g., x_2d from
        # x.reshape(b*m, k)), so each grad maps back to the original
        # kernel input via input_mappings.
        derivation_map = _extract_derivation_map(hf)
        final_grads_list: list[torch.Tensor] = [
            torch.zeros_like(t) for t in tensor_inputs
        ]
        mappings = cached_input_mappings
        for gi, m in enumerate(mappings):
            if gi >= len(result_list):
                break
            if m.tensor_name in knames:
                ki = knames.index(m.tensor_name)
                final_grads_list[ki] = final_grads_list[ki] + result_list[gi].reshape(
                    tensor_inputs[ki].shape
                )
            else:
                deriv = derivation_map.get(m.tensor_name)
                source = deriv[0] if deriv else None
                chain = deriv[1] if deriv else []
                if source and source in knames:
                    ki = knames.index(source)
                    final_grads_list[ki] = final_grads_list[
                        ki
                    ] + _apply_inverse_derivation(
                        result_list[gi], chain, tensor_inputs[ki].shape
                    )
        grads = (
            tuple(final_grads_list)
            if len(final_grads_list) > 1
            else final_grads_list[0]
        )

    if return_code:
        if bwd_bound._config is None:  # pyrefly: ignore [missing-attribute]
            bwd_bound._config = bwd_bound.env.config_spec.default_config()  # pyrefly: ignore [missing-attribute]
        triton_code: str = bwd_bound.to_triton_code(bwd_bound._config)  # pyrefly: ignore [missing-attribute]
        return grads, bwd_source, triton_code

    return grads
