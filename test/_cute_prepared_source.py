from __future__ import annotations

import ast


def expand_prepared_root_source(source: str) -> str:
    """Expose the fragment helper's primitive events for legacy-source comparisons.

    Only the root's SMEM fragment ABI is accepted. Every operand, issuer, phase,
    initialization bit and completion action survives expansion. The K count is
    checked against the original shared layout; all other source is retained.
    Descriptor and other prepared programs deliberately have no rewrite here.
    """
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }

    class Expand(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if ast.unparse(node) in (
                "from helion._compiler.cute.prepared_tcgen_edge import execute_prepared_continuation",
                "from helion._compiler.cute import prepared_tcgen_edge",
            ):
                return None
            return node

        def visit_Expr(self, node):
            if not isinstance(node.value, ast.Call):
                return node
            call = node.value
            name = ast.unparse(call.func)
            args = [ast.unparse(arg) for arg in call.args]
            if name == "prepared_tcgen_edge.execute_prepared_read":
                assert not call.keywords and len(args) == 9
                assert args[3:] == ["None", "None", "None", "False", "None", "0"]
                return ast.parse(
                    f"cute.copy({args[2]}, {args[0]}, {args[1]})\n"
                    "cute.arch.fence_view_async_tmem_load()"
                ).body
            if name == "prepared_tcgen_edge.execute_prepared_store":
                assert not call.keywords and len(args) == 5
                assert args[3:] == ["False", "None"]
                return ast.parse(f"cute.copy({args[2]}, {args[0]}, {args[1]})").body
            if name == "prepared_tcgen_edge.execute_prepared_store_completion":
                assert not call.keywords and args == ["False"]
                return ast.parse("cute.arch.fence_view_async_tmem_store()").body
            if name != "execute_prepared_continuation":
                return node
            assert not call.keywords and len(args) == 7
            assert args[4] == "0" and args[6] == "False"
            program = ast.literal_eval(call.args[0])
            ports, events, phases = call.args[1:4]
            assert all(
                isinstance(value, ast.Tuple) for value in (ports, events, phases)
            )
            assert len(ports.elts) == len(events.elts) == len(phases.elts) == 1
            port = ports.elts[0]
            assert isinstance(port, ast.Tuple) and len(port.elts) == 11
            a, b, acc, mma, bar, phase, tmem, base, atom, advance, swizzle = (
                ast.unparse(value) for value in port.elts
            )
            assert (tmem, atom, advance, swizzle) == ("False", "16", "()", "128")
            assert ast.unparse(events.elts[0]) == bar
            assert ast.unparse(phases.elts[0]) == phase
            assert mma.endswith("_mma")
            prefix = mma.removesuffix("_mma")
            assert (a, b, acc) == (f"{prefix}_ra", f"{prefix}_rb", f"{prefix}_acc")
            issuer = args[5]
            lines = []
            for instruction in program:
                opcode = instruction[0]
                if opcode == 4:
                    assert len(instruction) == 7
                    _, port_index, begin, end, initialized, commit, wait = instruction
                    assert port_index == begin == 0
                    assert all(
                        type(flag) is bool for flag in (initialized, commit, wait)
                    )
                    assert not wait or commit
                    lines += [
                        f"if {issuer}:",
                        f"    {mma}.set(tcgen05.Field.ACCUMULATE, {initialized})",
                    ]
                    kk = f"{prefix}_kk"
                    if base == "0":
                        layout = assignments[f"{prefix}_a_layout"]
                        assert isinstance(layout, ast.Call)
                        assert ast.unparse(layout.func) == "cute.tile_to_shape"
                        shape = ast.literal_eval(layout.args[1])
                        assert end * 16 == shape[1]
                        lines += [
                            f"    for {kk} in cutlass.range_constexpr(cute.size({a}, mode=[2])):"
                        ]
                    else:
                        assert end == 4 and base == "chain_k_half * 4"
                        local = f"{prefix}_local_kk"
                        lines += [
                            f"    for {local} in cutlass.range_constexpr(4):",
                            f"        {kk} = {base} + {local}",
                        ]
                    lines += [
                        f"        cute.gemm({mma}, {acc}, {a}[None, None, {kk}], {b}[None, None, {kk}], {acc})",
                        f"        {mma}.set(tcgen05.Field.ACCUMULATE, True)",
                    ]
                    if commit:
                        lines += [
                            "    with cute.arch.elect_one():",
                            f"        tcgen05.commit({bar})",
                        ]
                    if wait:
                        lines += [f"cute.arch.mbarrier_wait({bar}, {phase})"]
                else:
                    assert instruction in ((6, 0, False), (0, 0, False))
                    if opcode == 6:
                        lines += [
                            f"if {issuer}:",
                            "    with cute.arch.elect_one():",
                            f"        tcgen05.commit({bar})",
                        ]
                    else:
                        lines += [f"cute.arch.mbarrier_wait({bar}, {phase})"]
            return ast.parse("\n".join(lines)).body

    return ast.unparse(ast.fix_missing_locations(Expand().visit(tree)))


def assert_prepared_root_equivalent(before: str, after: str) -> str:
    expanded = expand_prepared_root_source(after)
    assert expanded == expand_prepared_root_source(before)
    return expanded
