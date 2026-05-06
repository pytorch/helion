from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from torch._functorch.aot_autograd import aot_module_simplified
import torch._functorch.config as functorch_config
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._inductor.decomposition import select_decomp_table
import torch.fx
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .._compiler.host_function import HostFunction
    from ..runtime.kernel import Kernel


@dataclass
class InputMapping:
    placeholder_name: str
    tensor_name: str
    fake_tensor: torch.Tensor | None


class GraphAnalyzer:
    """
    Analyzes forward Helion graph and extracts the pure computation subgraph.
    """

    def __init__(
        self,
        forward_graph: torch.fx.Graph,
        scalar_values: dict[str, object] | None = None,
        return_order: tuple[str, ...] | None = None,
    ) -> None:
        self.forward_graph = forward_graph
        self.scalar_values = scalar_values or {}
        # Host tensor names in kernel return order; aligns compute-graph
        # outputs with the user's grad_outs (which are passed in return order).
        self.return_order = return_order

    def _get_tensor_name(self, host_tensor_node: Node) -> str:
        target = host_tensor_node.target
        assert callable(target) and getattr(target, "__name__", "") == "_host_tensor"
        name = host_tensor_node.args[0]
        assert isinstance(name, str)
        return name

    def extract_computation_graph(
        self,
    ) -> tuple[torch.fx.Graph, list[InputMapping], list[tuple[int, ...]]]:
        """
        Extract computation subgraph.

        Returns:
            compute_graph: Pure PyTorch FX graph
            input_mappings: Load -> placeholder mappings
            output_shapes: Shapes of each compute graph output
        """
        compute_graph = torch.fx.Graph()
        node_map: dict[Node, Node] = {}
        input_mappings: list[InputMapping] = []

        tensor_to_placeholder: dict[str, Node] = {}
        tensor_current_value: dict[str, Node] = {}

        for node in self.forward_graph.nodes:
            if node.op != "call_function":
                continue

            target = node.target
            assert callable(target)
            target_name = target.__name__

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
                            placeholder_name=ph_name,
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
                if sym_name in self.scalar_values:
                    val = self.scalar_values[sym_name]
                    const_node = compute_graph.call_function(
                        torch.ops.aten.scalar_tensor.default,
                        (val,),  # pyrefly: ignore [bad-argument-type]
                    )
                    node_map[node] = const_node

            elif target_name == "subscript":
                # Only single-None unsqueeze is supported; multi-None or
                # non-slice indexing would silently lose information.
                tensor_node = node.args[0]
                assert isinstance(tensor_node, Node)
                index = node.args[1]
                assert isinstance(index, (list, tuple))
                index_seq = list(
                    index  # pyrefly: ignore [bad-argument-type]
                )
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
                    new_node = compute_graph.call_function(
                        torch.ops.aten.unsqueeze.default,
                        (node_map[tensor_node], none_pos[0]),
                    )
                    if node.meta:
                        new_node.meta = node.meta.copy()
                    node_map[node] = new_node
                elif len(none_pos) == 0:
                    node_map[node] = node_map[tensor_node]
                else:
                    raise exc.AutodiffNotSupported(
                        f"subscript with multiple None: {index_seq}"
                    )

            elif target_name != "_host_tensor":
                # Restore None placeholders from strip_unused_inputs (duplicate
                # arg coalescing) and reinject `_extra_args` kwargs.
                args = node.args
                first_node_arg = next((a for a in args if isinstance(a, Node)), None)
                if first_node_arg is not None:
                    args = tuple(first_node_arg if a is None else a for a in args)

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
        if self.return_order is not None:
            ordered_names: list[str] = []
            seen: set[str] = set()
            for name in self.return_order:
                if name in stored_outputs and name not in seen:
                    ordered_names.append(name)
                    seen.add(name)
            for name in stored_outputs:
                if name not in seen:
                    ordered_names.append(name)
        else:
            ordered_names = list(stored_outputs.keys())

        output_value_nodes = [stored_outputs[n] for n in ordered_names]
        outputs = [node_map[v] for v in output_value_nodes]
        compute_graph.output(tuple(outputs))

        output_shapes: list[tuple[int, ...]] = []
        for v in output_value_nodes:
            assert "val" in v.meta, (
                f"compute graph output {v.name} has no fake-tensor meta"
            )
            fake = v.meta["val"]
            output_shapes.append(tuple(fake.shape))
        return compute_graph, input_mappings, output_shapes


def differentiate_graph(
    compute_graph: torch.fx.Graph,
    input_tensors: tuple[torch.Tensor, ...],
) -> torch.fx.Graph:
    """
    Differentiate computation graph using AOT Autograd with full recomputation.

    Returns:
        backward_graph: FX graph for backward computation
    """
    example_inputs = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device, requires_grad=True)
        for t in input_tensors
    ]

    backward_graph: torch.fx.Graph | None = None

    def bw_compiler(
        gm: torch.fx.GraphModule,
        _aot_example_inputs: list[torch.Tensor],
    ) -> torch.fx.GraphModule:
        nonlocal backward_graph
        backward_graph = gm.graph
        return gm

    with functorch_config.patch(activation_memory_budget=0):
        compiled = aot_module_simplified(
            torch.fx.GraphModule({}, compute_graph),
            example_inputs,
            fw_compiler=lambda gm, _: gm,  # type: ignore[arg-type]
            bw_compiler=bw_compiler,  # type: ignore[arg-type]
            decompositions=select_decomp_table(),
            partition_fn=min_cut_rematerialization_partition,
        )

        example_out = compiled(*example_inputs)
        if isinstance(example_out, (list, tuple)):
            loss = sum(o.sum() for o in example_out)
        else:
            loss = example_out.sum()
        assert isinstance(loss, torch.Tensor)
        loss.backward()

    assert backward_graph is not None
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
    ) -> None:
        self.backward_graph = backward_graph
        self.grad_input_order = [m.tensor_name for m in input_mappings]

        # AOT Autograd numbers primals from 1.
        self.primal_to_name = {
            i + 1: m.tensor_name for i, m in enumerate(input_mappings)
        }

        self.tensor_shapes: dict[str, tuple[int, ...]] = {
            m.tensor_name: tuple(input_tensors[i].shape)
            for i, m in enumerate(input_mappings)
        }

        self.grad_out_shapes = grad_out_shapes
        self.num_grad_outs = len(grad_out_shapes)

        # Authoritative reduce dims from the forward IR (when supplied).
        self.forward_reduce_dims = forward_reduce_dims

    def _grad_out_index(self, param_name: str) -> int:
        """Return the grad_out index encoded in `param_name` (e.g.
        ``grad_out_3`` -> 3). Single-output kernels use the bare name
        ``grad_out`` which maps to index 0."""
        if param_name == "grad_out":
            return 0
        assert param_name.startswith("grad_out_")
        return int(param_name[len("grad_out_") :])

    def _param_ndim(self, param_name: str) -> int:
        """Get the ndim for an input parameter (grad_out or tensor input)."""
        return len(self._param_shape(param_name))

    def _param_shape(self, param_name: str) -> tuple[int, ...]:
        """Get the shape for an input parameter."""
        if param_name == "grad_out" or param_name.startswith("grad_out_"):
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

        if param_name == "grad_out" or param_name.startswith("grad_out_"):
            if param_ndim == len(non_reduced_dims):
                return list(non_reduced_dims)

        return _match_dims(param_shape, iter_shape)

    def _iter_tensor_name(self) -> str:
        """Name of the input that spans the iteration shape (highest rank)."""
        return max(self.grad_input_order, key=lambda n: len(self.tensor_shapes[n]))

    def _has_reductions(self) -> bool:
        """Check if the backward graph contains reduction operations."""
        for node in self.backward_graph.nodes:
            if node.op == "call_function":
                op_name = getattr(node.target, "_opname", None)
                if op_name in self._REDUCTION_OPS:
                    return True
        return False

    def _detect_reduced_dims(self, iter_shape: tuple[int, ...]) -> list[int]:
        """Return iter dims reduced from input to forward output.

        Prefers the forward IR's explicit reduce dims (authoritative). Falls
        back to shape-diffing iter_shape against grad_out; raises on ambiguity
        rather than guessing.
        """
        iter_ndim = len(iter_shape)
        if not self.grad_out_shapes:
            return []

        grad_shape = self.grad_out_shapes[0]
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

        if self.num_grad_outs == 1:
            grad_out_params = ["grad_out"]
        else:
            grad_out_params = [f"grad_out_{i}" for i in range(self.num_grad_outs)]
        input_params = [*grad_out_params, *self.grad_input_order]
        output_grad_names = [f"grad_{name}" for name in self.grad_input_order]

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
        self._use_reduction_kernel_path = (
            self._backward_has_reductions
            and iter_ndim >= 2
            and len(self._full_slice_dims) > 0
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

            if op_name == "unsqueeze":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    if self._use_reduction_kernel_path:
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
                                    node_to_var[node.name] = node_to_var[
                                        input_node.name
                                    ]
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

            result_var = f"{node.name}_val"
            node_to_var[node.name] = result_var

            arg_vars = [process_arg(arg) for arg in node.args]

            if op_name is not None and arg_vars:
                if hasattr(torch, op_name):
                    code = f"torch.{op_name}({', '.join(arg_vars)})"
                else:
                    tensor = arg_vars[0]
                    method_args = ", ".join(arg_vars[1:])
                    code = f"{tensor}.{op_name}({method_args})"
            elif op_name is not None:
                code = f"torch.{op_name}({', '.join(arg_vars)})"
            else:
                code = f"{node.target}({', '.join(arg_vars)})"

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

        assignments = []
        for i, out_node in enumerate(output_args_list):
            grad_name = f"grad_{self.grad_input_order[i]}"
            assert isinstance(out_node, Node)
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
            parse_stmt(f"{g} = torch.empty_like({g.replace('grad_', '')})")
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
                if tensor_ndim < iter_ndim:
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
                out_tensor = grad_name.replace("grad_", "")
                out_shape = self.tensor_shapes.get(out_tensor, ())
                if len(out_shape) == iter_ndim or out_shape == tiled_shape:
                    normal_outputs.append((grad_name, var_name))
                elif len(out_shape) > 0:
                    partial_name = f"{grad_name}_parts"
                    split_reduction_outputs.append((grad_name, var_name, partial_name))
                else:
                    normal_outputs.append((grad_name, var_name))

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
                    out_tensor = grad_name.replace("grad_", "")
                    out_ndim = len(self.tensor_shapes[out_tensor])
                    trailing_dims = ", ".join(
                        f"{out_tensor}.shape[{d}]" for d in range(out_ndim)
                    )
                    body.append(
                        parse_stmt(
                            f"{partial_name} = torch.empty("
                            f"[_num_blocks, {trailing_dims}],"
                            f" dtype={out_tensor}.dtype,"
                            f" device={out_tensor}.device)"
                        )
                    )

            for grad_name, var_name in normal_outputs:
                out_tensor = grad_name.replace("grad_", "")
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
                out_tensor = grad_name.replace("grad_", "")
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
                body.append(parse_stmt(f"{grad_name} = {partial_name}.sum(0)"))
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

            for grad_name, var_name in output_assignments:
                loop_body.append(
                    parse_stmt(f"{grad_name}[{', '.join(tile_vars)}] = {var_name}")
                )

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
    graph: torch.fx.Graph, *, iter_ndim: int | None = None
) -> set[int]:
    """Gather ``dim`` args from sum/amax/amin/mean nodes in ``graph``.

    Reads positional or keyword ``dim`` arguments. When ``iter_ndim`` is
    given, normalizes negative dims to non-negative via ``% iter_ndim``.
    """
    reduce_dims: set[int] = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        op_name = getattr(node.target, "_opname", None)
        if op_name not in FXToHelionConverter._REDUCTION_OPS:
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


def _extract_forward_reduce_dims(forward_graph: torch.fx.Graph) -> tuple[int, ...]:
    """Forward graph reduce dims, in iter-tile coordinates.

    The forward load tile has the same ndim as the iteration shape, so the
    raw ``dim`` args map onto iter dims with at most a sign flip. The
    consumer (`_detect_reduced_dims`) normalizes via ``% iter_ndim`` once
    iter_ndim is known.
    """
    return tuple(sorted(_collect_reduce_dims(forward_graph)))


def _validate_supported_reductions(forward_graph: torch.fx.Graph) -> None:
    """Reject reductions outside `_REDUCTION_OPS`.

    Their backward decompositions emit ops we don't codegen for (e.g.
    ``aten.slice.Tensor`` from ``prod``) and would otherwise fail late with
    cryptic errors.
    """
    for node in forward_graph.nodes:
        if node.op != "call_function":
            continue
        op_name = getattr(node.target, "_opname", None)
        if op_name in {
            "prod",
            "var",
            "std",
            "argmax",
            "argmin",
            "any",
            "all",
            "cumsum",
            "cumprod",
        }:
            raise exc.AutodiffNotSupported(
                f"reduction op {op_name!r} (only "
                f"{sorted(FXToHelionConverter._REDUCTION_OPS)} are supported)"
            )


def _extract_return_order(host_function: HostFunction) -> tuple[str, ...] | None:
    """Return host-tensor names in the order the kernel returns them.

    The user's ``grad_outs`` are in return order, but the forward graph stores
    values in first-store order — pairing them up needs the return order.
    Supports simple ``Name`` returns and ``expr.method(...)`` chains
    (e.g. ``inv_rms.reshape(-1, 1)``); returns None when the return value
    can't be reduced to plain names so callers fall back to store order.

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

    names: list[str] = []
    for elt in elements:
        name = _underlying_name(elt)
        if name is None:
            return None
        names.append(name)
    return tuple(names)


def _find_outer_return(body: list[ast.stmt]) -> ast.Return | None:
    """First ``Return`` reachable from ``body`` without crossing a nested
    function or lambda boundary."""
    for stmt in body:
        if isinstance(stmt, ast.Return):
            return stmt
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
    grad_outs: tuple[torch.Tensor, ...], target_ndims: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """Squeeze size-1 dims off each grad_out to match its target ndim.

    Needed when the forward reshapes outputs outside the tile loop (e.g.
    ``inv_rms.reshape(-1, 1)``); the generated bwd_fn was bound against the
    pre-reshape tile-loop ndims and must see the same ranks on every call.
    """
    return tuple(
        _squeeze_to_ndim(g, n, i)
        for i, (g, n) in enumerate(zip(grad_outs, target_ndims, strict=True))
    )


def _squeeze_to_ndim(g: torch.Tensor, target_ndim: int, idx: int) -> torch.Tensor:
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


def _underlying_name(expr: ast.expr) -> str | None:
    """Walk ``.method(...)`` / ``.attr`` chains down to the leftmost Name."""
    while True:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Call):
            expr = expr.func
        elif isinstance(expr, ast.Attribute):
            expr = expr.value
        else:
            return None


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

    cache_key = (autotune, autotune_effort)
    cache: dict[tuple[bool, str], tuple[Any, str, Any, tuple[int, ...]]] | None = (
        getattr(bound, "_backward_compiled_cache", None)
    )
    if cache is not None and cache_key in cache:
        bwd_fn, bwd_source, bwd_bound, target_ndims = cache[cache_key]
    else:
        from .._compiler.device_ir import ForLoopGraphInfo
        from .._compiler.device_ir import ReductionLoopGraphInfo
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
            elif isinstance(graph_info, ForLoopGraphInfo) and not isinstance(
                graph_info, ReductionLoopGraphInfo
            ):
                raise exc.AutodiffNotSupported("multiple tile loops")
        if root_graph_info is None:
            raise exc.AutodiffNotSupported("no root graph found")

        fwd_graph = root_graph_info.graph
        _validate_supported_reductions(fwd_graph)

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

        analyzer = GraphAnalyzer(
            fwd_graph,
            scalar_values=scalar_values,
            return_order=return_order,
        )
        compute_graph, input_mappings, compute_output_shapes = (
            analyzer.extract_computation_graph()
        )

        target_ndims = tuple(len(s) for s in compute_output_shapes)
        grad_outs = _canonicalize_grad_outs(grad_outs, target_ndims)

        backward_graph = differentiate_graph(compute_graph, tensor_inputs)

        converter = FXToHelionConverter(
            backward_graph=backward_graph,
            input_mappings=input_mappings,
            input_tensors=tensor_inputs,
            grad_out_shapes=tuple(g.shape for g in grad_outs),
            forward_reduce_dims=forward_reduce_dims,
        )
        bwd_source = converter.convert()

        # Stable on-disk path so `inspect.getsource` and tracebacks still
        # work after we return; write-temp-then-replace for safe concurrent
        # writers.
        from ..autotuner.local_cache import get_helion_cache_dir

        source_hash = hashlib.md5(
            bwd_source.encode(), usedforsecurity=False
        ).hexdigest()[:12]
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

        spec = importlib.util.spec_from_file_location(
            f"helion_bwd_{source_hash}", str(source_path)
        )
        assert spec is not None and spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "backward_kernel")

        bwd_fn = module.backward_kernel
        bwd_args = (*grad_outs, *tensor_inputs)
        bwd_bound = bwd_fn.bind(bwd_args)

        # Set on the bound kernel so the first call doesn't autotune.
        bwd_bound.settings.autotune_effort = autotune_effort
        if autotune:
            bwd_bound.autotune(bwd_args)

        if cache is None:
            cache = {}
            bound._backward_compiled_cache = cache  # pyrefly: ignore [missing-attribute]
        cache[cache_key] = (bwd_fn, bwd_source, bwd_bound, target_ndims)

    # Apply on every call (not just cache miss) — the cached `bwd_fn` was
    # compiled against canonicalized shapes; user grad_outs may still have
    # size-1 padding that needs squeezing.
    grad_outs = _canonicalize_grad_outs(grad_outs, target_ndims)
    result = bwd_fn(*grad_outs, *tensor_inputs)
    if isinstance(result, tuple):
        assert all(isinstance(r, torch.Tensor) for r in result)
        grads: torch.Tensor | tuple[torch.Tensor, ...] = (
            result if len(result) > 1 else result[0]
        )
    else:
        assert isinstance(result, torch.Tensor)
        grads = result

    if return_code:
        if bwd_bound._config is None:
            bwd_bound._config = bwd_bound.env.config_spec.default_config()
        triton_code: str = bwd_bound.to_triton_code(bwd_bound._config)
        return grads, bwd_source, triton_code

    return grads
