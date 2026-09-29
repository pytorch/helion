"""Explicit uniform-stream policy, independent of configuration and tiling."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import random_ops
from ...language.inline_asm_ops import inline_asm_elementwise
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from ...language.ref_tile import RefTile
    from ..inductor_lowering import CodegenState


def philox_assembly(packet: bool) -> str:
    """Philox4x32-10; reinterpret signed seeds as two unsigned keys."""
    outputs = 4 if packet else 1
    seed, offset = f"${outputs}", f"${outputs + 1}"
    lines = [
        "{",
        ".reg .b64 counter, low;",
        ".reg .b32 c0, c1, c2, c3, k0, k1, h0, h1, l0, l1, t0, t2, lane, word;",
        ".reg .s32 signed_word, sign, magnitude;",
        ".reg .f32 value;",
        ".reg .pred p0, p1, p2;",
        f"mov.b64 {{k0, k1}}, {seed};",
        f"shr.s64 counter, {offset}, 2;",
        "mov.b64 {c0, c1}, counter;",
        "mov.b32 c2, 0;",
        "mov.b32 c3, 0;",
    ]
    for round_index in range(10):
        lines += [
            "mul.hi.u32 h0, c2, 0xCD9E8D57;",
            "mul.lo.u32 l0, c2, 0xCD9E8D57;",
            "mul.hi.u32 h1, c0, 0xD2511F53;",
            "mul.lo.u32 l1, c0, 0xD2511F53;",
            "xor.b32 t0, h0, c1;",
            "xor.b32 t0, t0, k0;",
            "xor.b32 t2, h1, c3;",
            "xor.b32 t2, t2, k1;",
            "mov.b32 c0, t0;",
            "mov.b32 c1, l0;",
            "mov.b32 c2, t2;",
            "mov.b32 c3, l1;",
        ]
        if round_index != 9:
            lines += ["add.u32 k0, k0, 0x9E3779B9;", "add.u32 k1, k1, 0xBB67AE85;"]
    if not packet:
        lines += [
            f"and.b64 low, {offset}, 3;",
            "cvt.u32.u64 lane, low;",
            "setp.eq.u32 p0, lane, 0;",
            "setp.eq.u32 p1, lane, 1;",
            "setp.eq.u32 p2, lane, 2;",
            "selp.b32 word, c2, c3, p2;",
            "selp.b32 word, c1, word, p1;",
            "selp.b32 word, c0, word, p0;",
        ]
    for index in range(outputs):
        word = f"c{index}" if packet else "word"
        lines += [
            f"mov.b32 signed_word, {word};",
            "shr.s32 sign, signed_word, 31;",
            "xor.b32 magnitude, signed_word, sign;",
            "cvt.rn.f32.s32 value, magnitude;",
            f"mul.rn.f32 ${index}, value, 0f2FFFFFFF;",
        ]
    return "\n".join([*lines, "}"])


SCALAR_ASM = philox_assembly(False)
PACKET_ASM = philox_assembly(True)
SCALAR_CONSTRAINTS = "=f,l,l"
PACKET_CONSTRAINTS = "=f,=f,=f,=f,l,l"


def _host_seed(value: object) -> bool:
    from ..host_function import HostFunction

    if type(value) is int:
        return True
    if isinstance(value, torch.SymInt):
        origins = HostFunction.current().expr_to_origin
        return all(
            symbol in origins and origins[symbol].origin.is_host()
            for symbol in value._sympy_().free_symbols
        )
    return False


def rewrite_random_stream(graph: torch.fx.Graph) -> None:
    """Replace selected uniform calls before word0 decomposition can run."""
    env = CompileEnvironment.current()
    for node in random_ops._random_rewrite_nodes(graph):
        if node.target is not random_ops.rand:
            if env.settings.cute_rng_stream == "auto":
                continue
            raise exc.BackendUnsupported(
                "cute", "philox4 supports explicit uniform hl.rand only"
            )
        descriptors, shape_desc = random_ops._shape_rewrite_desc(node.args[0])
        seed_arg = node.args[1]
        seed_value = (
            seed_arg.meta["val"] if isinstance(seed_arg, torch.fx.Node) else seed_arg
        )
        stage_seed = _host_seed(seed_value)
        seed_desc = random_ops._add_rewrite_desc(descriptors, seed_arg)
        offsets_arg = node.args[3]
        offsets_desc = (
            random_ops._add_rewrite_desc(descriptors, offsets_arg)
            if offsets_arg is not None
            else None
        )

        def helper(
            *flat_args: object,
            shape_desc: tuple[random_ops._ArgDesc, ...] = tuple(shape_desc),
            seed_desc: random_ops._ArgDesc = seed_desc,
            offsets_desc: random_ops._ArgDesc | None = offsets_desc,
            stage_seed: bool = stage_seed,
        ) -> torch.Tensor:
            shape = [
                random_ops._resolve_shape_desc(flat_args, desc) for desc in shape_desc
            ]
            seed = random_ops._resolve_seed_desc(flat_args, seed_desc)
            offsets = random_ops._resolve_offsets_desc(flat_args, offsets_desc)
            if offsets is None:
                offsets = random_ops._explicit_offset_from_shape(
                    shape, device=env.device
                )
            offsets = random_ops._as_int64_scalar(offsets, device=env.device)
            seed = random_ops._as_int64_scalar(
                0 if stage_seed else seed, device=env.device
            )
            return inline_asm_elementwise(
                SCALAR_ASM,
                SCALAR_CONSTRAINTS,
                (seed, offsets),
                dtype=torch.float32,
                is_pure=True,
                pack=1,
            )

        replacement = random_ops._trace_rewrite_subgraph(
            graph, node, helper, descriptors
        )
        assert isinstance(replacement, torch.fx.Node)
        assert replacement.target is inline_asm_elementwise
        seed_value = (
            seed_arg.meta["val"] if isinstance(seed_arg, torch.fx.Node) else seed_arg
        )
        if stage_seed:
            replacement.meta["cute_philox_host_seed"] = seed_value
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        env.config_spec.cute_rng_packet_enabled = True


def stage_explicit_seed(state: CodegenState, seed: int | torch.SymInt) -> ast.expr:
    """Create a fresh graph-visible seed argument at the original host launch."""
    from ..device_function import TensorArg
    from ..host_function import HostFunction

    fn = state.device_function
    env = CompileEnvironment.current()
    host_seed = HostFunction.current().literal_expr(seed)
    tensor_inputs = [arg for arg in fn.arguments if isinstance(arg, TensorArg)]
    if tensor_inputs:
        device = tensor_inputs[0].host_str()
    else:
        # Shape-only inputs may have no device pointer argument yet. Their
        # original host argument still proves the allocation device and is
        # available before every generated launch.
        origins = [
            origin
            for tensor, origin in HostFunction.current().tensor_to_origin.items()
            if origin.is_argument() and tensor.device == env.device
        ]
        if origins:
            device = origins[0].host_str()
        else:
            # Host locals replace ArgumentOrigin during type propagation, and
            # an RNG node may precede every tensor load/store in this graph.
            # _host_tensor nodes still name tensors available at this launch,
            # including outputs of kernels with no tensor inputs. Do not scan
            # arbitrary host locals, which may belong to a different launch.
            assert state.fx_node is not None
            host_tensors = [
                node
                for node in state.fx_node.graph.nodes
                if node.target is _tracing_ops._host_tensor
                and isinstance(value := node.meta.get("val"), torch.Tensor)
                and value.device == env.device
            ]
            if not host_tensors:
                raise exc.BackendUnsupported(
                    "cute", "explicit seed staging needs a tensor input device"
                )
            device = cast("str", host_tensors[0].args[0])
    with env.fake_mode:
        fake = torch.empty((1,), dtype=torch.int64, device=env.device)
    arg = TensorArg(
        fn.new_var("explicit_rng_seed"),
        fake,
        f"_cute_stage_explicit_seed({host_seed}, {device})",
    )
    fn.arguments.append(arg)
    fn.cute_state.explicit_rng_seed_names.add(arg.name)
    fn._tensor_args[fake] = arg
    statement = statement_from_string(
        "from helion.runtime.cute.rng import stage_explicit_seed as _cute_stage_explicit_seed"
    )
    if not any(
        ast.dump(value) == ast.dump(statement)
        for value in state.codegen.module_statements
    ):
        state.codegen.module_statements.append(statement)
    return cast("ast.expr", expr_from_string(f"{arg.name}[0]"))


def reference_override(
    function: object, args: tuple[object, ...], *, strict: bool = True
) -> tuple[bool, object]:
    """Use the pre-existing tensor oracle, independently of generated assembly."""
    if function is random_ops.rand:
        from ..rng_utils import philox_rand4x_ref

        shape, seed, device, offsets = cast(
            "tuple[list[int | RefTile], int | torch.Tensor, torch.device | None, torch.Tensor | None]",
            args,
        )
        env = CompileEnvironment.current()
        rng_device = env.device if device is None else device
        if offsets is None:
            processed_shape, offsets = random_ops._ref_rng_shape_and_offset(
                shape, device=rng_device
            )
        else:
            processed_shape = offsets.shape
        values = philox_rand4x_ref(seed, torch.div(offsets, 4, rounding_mode="floor"))
        lane = offsets & 3
        result = values[3]
        for index in (2, 1, 0):
            result = torch.where(lane == index, values[index], result)
        return True, result.reshape(processed_shape).to(device=rng_device)
    if strict and function in (random_ops.rand4x, random_ops.randint):
        raise exc.BackendUnsupported(
            "cute", "philox4 supports explicit uniform hl.rand only"
        )
    return False, None
