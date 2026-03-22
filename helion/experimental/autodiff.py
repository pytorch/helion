from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import pathlib
import tempfile
from typing import TYPE_CHECKING
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
    ) -> None:
        self.forward_graph = forward_graph
        self.scalar_values = scalar_values or {}

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

        # Track current value of each tensor (None = need placeholder for original)
        tensor_to_placeholder: dict[str, Node] = {}
        tensor_current_value: dict[str, Node] = {}

        # Process nodes in order to preserve load/store semantics
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

                # Check if there's a prior store to this tensor
                if tensor_name in tensor_current_value:
                    # Load after store: use the stored value
                    stored_value_node = tensor_current_value[tensor_name]
                    node_map[node] = node_map[stored_value_node]
                elif tensor_name in tensor_to_placeholder:
                    # Another load of same tensor (before any store): reuse placeholder
                    node_map[node] = tensor_to_placeholder[tensor_name]
                else:
                    # First load of this tensor: create placeholder
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
                # Map multi-buffer lowering to its data input
                data_inputs = node.args[0]
                assert isinstance(data_inputs, (list, tuple)) and len(data_inputs) >= 1
                assert isinstance(data_inputs[0], Node)
                node_map[node] = node_map[data_inputs[0]]

            elif target_name == "_mask_to":
                assert isinstance(node.args[0], Node)
                node_map[node] = node_map[node.args[0]]

            elif target_name == "_get_symnode":
                # Resolve scalar parameter (e.g., eps) to concrete value
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
                # Convert subscript indexing (e.g., tensor[:, None]) to unsqueeze
                tensor_node = node.args[0]
                assert isinstance(tensor_node, Node)
                index = node.args[1]
                assert isinstance(index, (list, tuple))
                none_pos = [
                    i
                    for i, idx in enumerate(
                        index  # pyrefly: ignore [bad-argument-type]
                    )
                    if idx is None
                ]
                if len(none_pos) == 1:
                    new_node = compute_graph.call_function(
                        torch.ops.aten.unsqueeze.default,
                        (node_map[tensor_node], none_pos[0]),
                    )
                    if node.meta:
                        new_node.meta = node.meta.copy()
                    node_map[node] = new_node
                else:
                    node_map[node] = node_map[tensor_node]

            elif target_name != "_host_tensor":
                # Helion's strip_unused_inputs replaces duplicate node args with None when they map to the same input
                # buffer (e.g., val * val -> mul(val, None)). We restore the real arg for differentiate_graph
                args = node.args
                first_node_arg = next((a for a in args if isinstance(a, Node)), None)
                if first_node_arg is not None:
                    args = tuple(first_node_arg if a is None else a for a in args)

                # Restore _extra_args (e.g., mean.dim) into positional args
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

                new_args = map_arg(args, node_map.get)
                new_kwargs = map_arg(kwargs, node_map.get)
                target = node.target
                assert callable(target)
                new_node = compute_graph.call_function(target, new_args, new_kwargs)
                if node.meta:
                    new_node.meta = node.meta.copy()
                node_map[node] = new_node

        input_tensor_names = set(tensor_to_placeholder.keys())
        output_value_nodes = [
            v for t, v in tensor_current_value.items() if t not in input_tensor_names
        ]
        outputs = [node_map[v] for v in output_value_nodes]
        compute_graph.output(tuple(outputs))

        output_shapes = []
        for v in output_value_nodes:
            if "val" in v.meta:
                fake = v.meta["val"]
                output_shapes.append(tuple(fake.shape))
            else:
                output_shapes.append(())
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

    # Capture backward graph via compiler callback
    backward_graph: torch.fx.Graph | None = None

    def bw_compiler(
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
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

    def __init__(
        self,
        backward_graph: torch.fx.Graph,
        input_mappings: list[InputMapping],
        input_tensors: tuple[torch.Tensor, ...],
        grad_out_shapes: tuple[tuple[int, ...], ...],
    ) -> None:
        self.backward_graph = backward_graph
        self.grad_input_order = [m.tensor_name for m in input_mappings]

        # Map primal index (1-based from AOT Autograd) to tensor name
        self.primal_to_name = {
            i + 1: m.tensor_name for i, m in enumerate(input_mappings)
        }

        # Map tensor name to concrete shape from real input tensors
        self.tensor_shapes: dict[str, tuple[int, ...]] = {
            m.tensor_name: tuple(input_tensors[i].shape)
            for i, m in enumerate(input_mappings)
        }

        # Shapes of grad_out tensors (one per forward output)
        self.grad_out_shapes = grad_out_shapes
        self.num_grad_outs = len(grad_out_shapes)

    _REDUCTION_OPS: ClassVar[set[str]] = {"sum", "amax", "amin", "mean"}

    def _param_ndim(self, param_name: str) -> int:
        """Get the ndim for an input parameter (grad_out or tensor input)."""
        if param_name == "grad_out":
            return len(self.grad_out_shapes[0])
        if param_name.startswith("grad_out_"):
            idx = int(param_name.split("_")[-1])
            return len(self.grad_out_shapes[idx])
        return len(self.tensor_shapes[param_name])

    def _param_shape(self, param_name: str) -> tuple[int, ...]:
        """Get the shape for an input parameter."""
        if param_name == "grad_out":
            return self.grad_out_shapes[0]
        if param_name.startswith("grad_out_"):
            idx = int(param_name.split("_")[-1])
            return self.grad_out_shapes[idx]
        return self.tensor_shapes.get(param_name, ())

    def _map_param_to_iter_dims(
        self, param_name: str, iter_shape: tuple[int, ...], non_reduced_dims: list[int]
    ) -> list[int]:
        """Map a parameter's dimensions to iter_shape dimension indices.

        Uses the known non-reduced dimensions to correctly match params that
        have fewer dims than iter_shape. This avoids the ambiguity of size-based
        matching when dimensions have equal sizes.

        For a grad_out with shape matching the output, its dims correspond
        exactly to the non-reduced dims of iter_shape.

        For other params with fewer dims, uses size-based matching against
        the non-reduced dims, falling back to leading alignment.
        """
        param_shape = self._param_shape(param_name)
        param_ndim = len(param_shape)
        iter_ndim = len(iter_shape)

        if param_ndim >= iter_ndim:
            return list(range(iter_ndim))

        # If this is a grad_out param, its dims map directly to non_reduced_dims
        if param_name == "grad_out" or param_name.startswith("grad_out_"):
            if param_ndim == len(non_reduced_dims):
                return list(non_reduced_dims)

        # For other params, try matching their shape to the non-reduced dims of iter
        non_reduced_shape = tuple(iter_shape[d] for d in non_reduced_dims)
        if param_shape == non_reduced_shape:
            return list(non_reduced_dims)

        # Try matching against the full iter_shape by size
        return _match_dims(param_shape, iter_shape)

    def _has_reductions(self) -> bool:
        """Check if the backward graph contains reduction operations."""
        for node in self.backward_graph.nodes:
            if node.op == "call_function":
                op_name = getattr(node.target, "_opname", None)
                if op_name in self._REDUCTION_OPS:
                    return True
        return False

    def _detect_reduced_dims(self, iter_shape: tuple[int, ...]) -> list[int]:
        """Detect which dimensions of iter_shape were reduced in the forward pass.

        Compares iter_shape against grad_out_shapes using subsequence matching
        to find which dimensions are "missing" from the output. This works
        correctly even when dimensions have equal sizes.

        For example:
            iter_shape=(8, 16, 32), grad_out=(8, 32) → reduced_dims=[1]
            iter_shape=(64, 32), grad_out=(64,) → reduced_dims=[1]
            iter_shape=(32, 32), grad_out=(32,) → uses backward graph analysis
        """
        iter_ndim = len(iter_shape)
        if not self.grad_out_shapes:
            return []

        # Use the first grad_out to determine which dims were reduced
        grad_shape = self.grad_out_shapes[0]
        grad_ndim = len(grad_shape)

        if grad_ndim >= iter_ndim:
            return []

        # Subsequence matching: find which iter_shape positions are NOT
        # needed to form grad_shape. Those unmatched positions are the
        # reduced dims. When sizes are unique this gives one answer;
        # when sizes repeat (e.g., [32, 32] -> [32]) it returns all
        # possible matchings so we can disambiguate later.
        def _find_reduced(grad_pos: int, iter_pos: int) -> list[list[int]]:
            """Return all possible lists of reduced dim indices."""
            if grad_pos == grad_ndim:
                # All grad dims matched; remaining iter dims are reduced
                return [list(range(iter_pos, iter_ndim))]
            if iter_pos == iter_ndim:
                # Ran out of iter dims before matching all grad dims
                return []
            results = []
            if iter_shape[iter_pos] == grad_shape[grad_pos]:
                # iter_pos matches grad_pos — not reduced
                results.extend(_find_reduced(grad_pos + 1, iter_pos + 1))
            # iter_pos doesn't match (or we also try skipping it) — reduced
            for rest in _find_reduced(grad_pos, iter_pos + 1):
                results.append([iter_pos, *rest])
            return results

        all_reduced = _find_reduced(0, 0)

        if len(all_reduced) == 1:
            return all_reduced[0]

        if len(all_reduced) == 0:
            # Fallback: assume trailing dims are reduced
            return list(range(grad_ndim, iter_ndim))

        # Multiple valid matchings (e.g., [32, 32] -> [32] could reduce
        # dim 0 or dim 1). Use backward graph reduction dim hints.
        bwd_reduce_dims = self._get_backward_reduce_dims(iter_ndim)
        if bwd_reduce_dims:
            # The backward graph's reduction dims are relative to the tile,
            # which loads the full iter_shape. Pick the matching that aligns
            # with those reduction dims.
            for reduced in all_reduced:
                if set(reduced) == set(bwd_reduce_dims):
                    return reduced

        # Default: prefer reducing later dims (last-dim reduction is most common)
        return all_reduced[-1]

    def _get_backward_reduce_dims(self, iter_ndim: int) -> list[int]:
        """Extract reduction dimension indices from backward graph ops.

        Looks at the dim arguments of sum/amax/amin/mean operations in the
        backward graph to determine which dimensions are being reduced.
        Normalizes negative indices to positive using iter_ndim.
        """
        reduce_dims: set[int] = set()
        for node in self.backward_graph.nodes:
            if node.op == "call_function":
                op_name = getattr(node.target, "_opname", None)
                if op_name in self._REDUCTION_OPS and len(node.args) >= 2:
                    dim_arg = node.args[1]
                    if isinstance(dim_arg, (list, tuple)):
                        for d in dim_arg:
                            if isinstance(d, int):
                                reduce_dims.add(d % iter_ndim)
                    elif isinstance(dim_arg, int):
                        reduce_dims.add(dim_arg % iter_ndim)
        return sorted(reduce_dims)

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

        # Generate code components
        if self.num_grad_outs == 1:
            grad_out_params = ["grad_out"]
        else:
            grad_out_params = [f"grad_out_{i}" for i in range(self.num_grad_outs)]
        input_params = [*grad_out_params, *self.grad_input_order]
        output_grad_names = [f"grad_{name}" for name in self.grad_input_order]

        self._backward_has_reductions = self._has_reductions()

        iter_tensor_name = output_grad_names[0].replace("grad_", "")
        iter_ndim = len(self.tensor_shapes[iter_tensor_name])
        iter_shape = self.tensor_shapes[iter_tensor_name]

        # Dimension classification (4 related concepts):
        #   _forward_reduced_dims: dims missing from grad_out (forward reduction)
        #   _full_slice_dims: dims needing ':' in tile loop (for backward reductions)
        #   _non_reduced_dims: dims present in grad_out (for loading grad_out)
        #   _tiled_dims: dims we iterate over (= iter_dims - full_slice_dims)
        #
        # These differ when backward has reductions but forward doesn't
        # (e.g., softmax): _forward_reduced_dims=[], _full_slice_dims=[last].

        self._forward_reduced_dims = self._detect_reduced_dims(iter_shape)

        if self._forward_reduced_dims:
            self._full_slice_dims = self._forward_reduced_dims
        elif self._backward_has_reductions and iter_ndim >= 2:
            # No forward reduction, but backward has reductions.
            # Find which dims the backward reduces on and use those.
            bwd_dims = self._get_backward_reduce_dims(iter_ndim)
            # If backward reduces on all dims, pick the last one
            # (most common case: row-wise softmax reduces last dim)
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
        """Map backward graph node name to generated variable name."""
        if node_name.startswith("primals_"):
            idx = int(node_name.split("_")[1])
            return f"{self.primal_to_name[idx]}_tile"
        if node_name.startswith("tangents_"):
            if self.num_grad_outs == 1:
                return "grad_out_tile"
            # tangents_1, tangents_2, ... → grad_out_0_tile, grad_out_1_tile, ...
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

        # Map all placeholders (primals_ and tangents_) to variable names
        for ph in placeholders:
            node_to_var[ph.name] = self._get_var_name(ph.name)

        # Generate code for each computation node
        for node in computations:
            target = node.target
            op_name = getattr(target, "_opname", None)

            # Skip identity ops - just alias to input variable
            if op_name in {"detach", "alias"}:
                if node.args:
                    input_node = node.args[0]
                    assert isinstance(input_node, Node)
                    node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            # Skip scalar_tensor - will inline the literal value
            if op_name == "scalar_tensor":
                if node.args:
                    node_to_var[node.name] = repr(node.args[0])
                    continue

            # Convert unsqueeze to reshape or subscript indexing
            if op_name == "unsqueeze":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    if self._use_reduction_kernel_path:
                        result_var = f"{node.name}_val"
                        node_to_var[node.name] = result_var
                        input_var = node_to_var[input_node.name]
                        in_ndim = (
                            input_node.meta["val"].ndim
                            if "val" in input_node.meta
                            else 1
                        )
                        dim = node.args[1] if len(node.args) > 1 else -1
                        if in_ndim <= 1:
                            out_ndim = in_ndim + 1
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

            # Skip expand — broadcasting handles this
            if op_name == "expand":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            # Handle view/reshape
            if op_name == "view":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    if self._use_reduction_kernel_path and len(node.args) >= 2:
                        target_shape = node.args[1]
                        if isinstance(target_shape, (list, tuple)):
                            in_ndim = (
                                input_node.meta["val"].ndim
                                if "val" in input_node.meta
                                else len(target_shape)
                            )
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
                                # Multiple tiled dims — use subscript indexing
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

            # Handle squeeze
            if op_name == "squeeze":
                if node.args and isinstance(node.args[0], Node):
                    input_node = node.args[0]
                    if self._use_reduction_kernel_path:
                        result_var = f"{node.name}_val"
                        node_to_var[node.name] = result_var
                        input_var = node_to_var[input_node.name]
                        out_ndim = node.meta["val"].ndim if "val" in node.meta else 1
                        if out_ndim <= 1:
                            lines.append(f"{result_var} = {input_var}.reshape(-1)")
                        else:
                            # Multi-dim: use squeeze(dim) if dim arg exists,
                            # else fall back to reshape
                            dim = node.args[1] if len(node.args) > 1 else None
                            if dim is not None:
                                lines.append(
                                    f"{result_var} = {input_var}.squeeze({dim})"
                                )
                            else:
                                lines.append(f"{result_var} = {input_var}.squeeze(-1)")
                    else:
                        node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            result_var = f"{node.name}_val"
            node_to_var[node.name] = result_var

            arg_vars = [process_arg(arg) for arg in node.args]

            # Generate op code: torch function or tensor method
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
        """
        Map backward graph outputs to gradient variable assignments.
        The backward graph returns gradients in the same order as forward inputs.
        Uses node_to_var to resolve aliases from skipped ops (unsqueeze/expand).
        """
        if output_node is None:
            return []

        # FX output node stores return values in args[0]
        output_args = output_node.args[0]
        if isinstance(output_args, (list, tuple)):
            output_args_list = list(output_args)
        else:
            output_args_list = [output_args]

        # Pair each output with its corresponding gradient name
        assignments = []
        for i, out_node in enumerate(output_args_list):
            grad_name = f"grad_{self.grad_input_order[i]}"
            assert isinstance(out_node, Node)
            # Use node_to_var which includes aliases from skipped ops
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
        """Build the complete Helion kernel source code using AST."""

        # Iteration shape comes from the first output gradient's tensor
        iter_var = output_grad_names[0]
        iter_tensor_name = iter_var.replace("grad_", "")
        iter_shape = self.tensor_shapes[iter_tensor_name]
        iter_ndim = len(iter_shape)

        needs_broadcast = self._needs_broadcast
        backward_has_reductions = self._backward_has_reductions

        def parse_expr(code: str) -> ast.expr:
            return ast.parse(code, mode="eval").body

        def parse_stmt(code: str) -> ast.stmt:
            return ast.parse(code, mode="exec").body[0]

        # Build imports
        imports: list[ast.stmt] = [
            ast.Import(names=[ast.alias(name="torch", asname=None)]),
            ast.Import(names=[ast.alias(name="helion", asname=None)]),
            ast.ImportFrom(
                module="helion",
                names=[ast.alias(name="language", asname="hl")],
                level=0,
            ),
        ]

        # Build function parameters: (grad_out: torch.Tensor, x: torch.Tensor, ...)
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

        # Return type: torch.Tensor or tuple[torch.Tensor, ...]
        multi_output = len(output_grad_names) > 1
        return_annotation = parse_expr(
            "tuple[torch.Tensor, ...]" if multi_output else "torch.Tensor"
        )

        # Function body before loop: grad_x = torch.empty_like(x)
        body: list[ast.stmt] = [
            parse_stmt(f"{g} = torch.empty_like({g.replace('grad_', '')})")
            for g in output_grad_names
        ]

        # Loop body: loads → computation → stores
        loop_body: list[ast.stmt] = []

        if (
            backward_has_reductions
            and iter_ndim >= 2
            and len(self._full_slice_dims) > 0
        ):
            # Reduction-in-backward: iterate tiled dims, ':' for full-slice dims.
            # Handles amax/softmax backward where computation requires reductions.
            full_slice_dims = set(self._full_slice_dims)
            tiled_dims = self._tiled_dims
            n_tiled = len(tiled_dims)

            dim_vars = [f"_dim_{i}" for i in range(n_tiled)]
            tile_vars = [f"tile_{i}" for i in range(n_tiled)]

            # Shape unpacking: e.g., "_dim_0, _, _dim_1 = grad_x.shape"
            shape_parts = []
            dim_idx = 0
            for d in range(iter_ndim):
                if d in full_slice_dims:
                    shape_parts.append("_")
                else:
                    shape_parts.append(dim_vars[dim_idx])
                    dim_idx += 1
            body.append(parse_stmt(f"{', '.join(shape_parts)} = {iter_var}.shape"))

            # e.g., iter_shape=(8,16,32) with full_slice={1} → [tile_0, ':', tile_1]
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

                    # Param only spans full-slice dims (e.g., weight [N])
                    if all(d in full_slice_dims for d in param_iter_dims):
                        load_expr = f"{p}[:]"
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
                        if not indices:
                            indices = [tile_vars[0]]
                        load_expr = f"{p}[{', '.join(indices)}]"
                else:
                    load_expr = f"{p}[{', '.join(full_indices)}]"
                loop_body.append(parse_stmt(f"{p}_tile = {load_expr}"))

            for line in computation_lines:
                loop_body.append(parse_stmt(line))

            # Split-reduction: outputs spanning all iter dims or exactly the
            # tiled dims can be stored directly. Others need partial accumulation.
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

            # Split-reduction: register_block_size + partial buffers
            use_block_size = len(split_reduction_outputs) > 0
            block_var = "_m_block"
            if use_block_size:
                # Allocate partial buffers
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
                else:
                    store_idx = ", ".join(tile_vars)
                loop_body.append(parse_stmt(f"{grad_name}[{store_idx}] = {var_name}"))
            for _grad_name, var_name, partial_name in split_reduction_outputs:
                loop_body.append(
                    parse_stmt(f"{partial_name}[{tile_vars[0]}.id, :] = {var_name}")
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

            # Sum partial buffers after the loop
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
            # Simple path: all inputs have the same ndim
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

        # Return statement
        if multi_output:
            return_value = ast.Tuple(
                elts=[ast.Name(id=g, ctx=ast.Load()) for g in output_grad_names],
                ctx=ast.Load(),
            )
        else:
            return_value = ast.Name(id=output_grad_names[0], ctx=ast.Load())
        body.append(ast.Return(value=return_value))

        # Function definition with @helion.kernel() decorator
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
    """Match a smaller tensor's dims to an iteration shape's dims by size.

    Returns a list of iter_shape indices that each param dim maps to.
    E.g., param [8, 32] with iter [8, 16, 32] → [0, 2].
    Falls back to leading alignment if sizes don't uniquely match.
    """
    param_ndim = len(param_shape)
    iter_ndim = len(iter_shape)

    if param_ndim >= iter_ndim:
        return list(range(iter_ndim))

    # Try to match each param dim to an iter dim by size
    mapping: list[int] = []
    used: set[int] = set()
    for p_dim in range(param_ndim):
        p_size = param_shape[p_dim]
        matched = False
        for i_dim in range(iter_ndim):
            if i_dim not in used and iter_shape[i_dim] == p_size:
                mapping.append(i_dim)
                used.add(i_dim)
                matched = True
                break
        if not matched:
            # Ambiguous — fall back to leading alignment
            return list(range(param_ndim))

    return mapping


def _resolve_scalar_values(
    kernel: Kernel[object],
    inputs: tuple[torch.Tensor, ...],
    fwd_graph: torch.fx.Graph,
) -> dict[str, object]:
    """Resolve _get_symnode names to concrete scalar values from kernel args."""
    import inspect

    all_args = kernel.normalize_args(*inputs)

    sig = inspect.signature(kernel.fn)
    param_names = list(sig.parameters.keys())
    param_to_value: dict[str, object] = {}
    for i, name in enumerate(param_names):
        if i < len(all_args) and not isinstance(all_args[i], torch.Tensor):
            param_to_value[name] = all_args[i]

    # Symnode names (e.g. 'zuf0') are auto-generated; match them
    # to concrete values via BoundArguments order.
    scalar_values: dict[str, object] = {}
    bound_params = None
    for bound_kernel in kernel._bound_kernels.values():
        if bound_kernel.host_function is not None:
            bound_params = bound_kernel.host_function.params
            break

    if bound_params is not None:
        for param_name, fake_val in bound_params.arguments.items():
            if param_name in param_to_value and not isinstance(fake_val, torch.Tensor):
                sym_name = str(fake_val)
                scalar_values[sym_name] = param_to_value[param_name]

    return scalar_values


def backward(
    kernel: Kernel[object],
    grad_out: torch.Tensor | tuple[torch.Tensor, ...],
    *inputs: torch.Tensor,
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

    Args:
        kernel: A @helion.kernel decorated function (must be called once first)
        grad_out: Gradient of loss w.r.t. kernel output. For multi-output kernels,
            pass a tuple of gradient tensors (one per output).
        *inputs: The original inputs to the kernel (in the same order as forward)
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

    bound = kernel.bind(inputs)
    if bound._config is None:
        bound._config = bound.env.config_spec.default_config()

    if bound._backward_compiled is not None:
        bwd_fn, bwd_source, bwd_bound = bound._backward_compiled
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

        scalar_values = _resolve_scalar_values(kernel, inputs, fwd_graph)

        analyzer = GraphAnalyzer(fwd_graph, scalar_values=scalar_values)
        compute_graph, input_mappings, compute_output_shapes = (
            analyzer.extract_computation_graph()
        )

        # Squeeze trailing size-1 dims from grad_outs to match
        # compute graph output ndims (forward may reshape outside tile loop)
        reshaped_grad_outs = []
        for i, g in enumerate(grad_outs):
            if i < len(compute_output_shapes):
                target_ndim = len(compute_output_shapes[i])
                if g.ndim > target_ndim:
                    extra_dims = g.shape[target_ndim:]
                    if all(d == 1 for d in extra_dims):
                        g = g.reshape(g.shape[:target_ndim])
            reshaped_grad_outs.append(g)
        grad_outs = tuple(reshaped_grad_outs)

        backward_graph = differentiate_graph(compute_graph, inputs)

        converter = FXToHelionConverter(
            backward_graph=backward_graph,
            input_mappings=input_mappings,
            input_tensors=inputs,
            grad_out_shapes=tuple(g.shape for g in grad_outs),
        )
        bwd_source = converter.convert()

        with tempfile.TemporaryDirectory(prefix="helion_bwd_") as cache_dir:
            source_hash = hashlib.md5(
                bwd_source.encode(), usedforsecurity=False
            ).hexdigest()[:12]
            temp_path = pathlib.Path(cache_dir) / f"helion_bwd_{source_hash}.py"

            temp_path.write_text(bwd_source)

            spec = importlib.util.spec_from_file_location(
                f"helion_bwd_{source_hash}", str(temp_path)
            )
            assert spec is not None and spec.loader is not None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            assert hasattr(module, "backward_kernel")

            bwd_fn = module.backward_kernel
            bwd_args = (*grad_outs, *inputs)
            bwd_bound = bwd_fn.bind(bwd_args)

            # Determine autotune_effort: use 'quick' when autotuning, 'none' otherwise
            if autotune_effort is None:
                autotune_effort = "quick" if autotune else "none"

            # Set autotune_effort to prevent automatic autotuning on first call
            bwd_bound.settings.autotune_effort = autotune_effort
            if autotune:
                bwd_bound.autotune(bwd_args)
            bound._backward_compiled = (bwd_fn, bwd_source, bwd_bound)

    result = bwd_fn(*grad_outs, *inputs)
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
