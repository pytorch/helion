"""The addresses a kernel's PTX stores to, as numbers, without a device.

Two compiles of one kernel that differ in a few instructions number their
registers differently from the first difference on, and one may fold an
offset into a store's operand where the other adds it to a register. To say
whether both store to the same places, ``store_addresses`` evaluates the
address of every matching store: through the integer instructions that
define its register, exactly, and over arbitrary values for everything else
(a parameter, a special register, a loaded word), each named by the
structure of its own definition and not by its register. A register with
several definitions, a loop's or a branch's, is not followed: all such
registers stand for one value. Equal lists from two kernels, for several
choices of those values, are equal address arithmetic over the registers
with one definition; which register with several definitions a store reads
is not compared.
"""

from __future__ import annotations

import hashlib
import re

_REGISTER = re.compile(r"%[A-Za-z_][\w.]*")
_INSTRUCTION = re.compile(r"^\s*(?:@!?%\w+\s+)?([a-z][\w.:]*)\s+(.*?);\s*$")
_MASKS = {32: (1 << 32) - 1, 64: (1 << 64) - 1}


def _width(opcode: str) -> int:
    return 64 if opcode.endswith("64") else 32


def _signed(value: int, bits: int) -> int:
    value &= _MASKS[bits]
    return value - (1 << bits) if value >> (bits - 1) else value


class _Kernel:
    """The single definitions of a kernel's registers."""

    def __init__(self, ptx: str) -> None:
        self.definitions: dict[str, tuple[str, list[str]]] = {}
        repeated: set[str] = set()
        for line in ptx.splitlines():
            match = _INSTRUCTION.match(line)
            if match is None or match.group(1).startswith(("st.", "bra", "ret")):
                continue
            opcode, rest = match.groups()
            operands = [operand.strip() for operand in rest.split(",")]
            target = operands[0]
            if not _REGISTER.fullmatch(target):
                # A vector of results (a wide load, a split): each is a
                # value of its own, named by the instruction and its place.
                names = _REGISTER.findall(rest.split("}")[0]) if "{" in rest else []
                tail = rest.split("}", 1)[1] if "}" in rest else ""
                for place, name in enumerate(names):
                    self._define(name, f"{opcode}#{place}", [tail], repeated)
                continue
            self._define(target, opcode, operands[1:], repeated)
        for name in repeated:
            # A register with several definitions is a loop's or a branch's:
            # its value is not this module's to follow.
            self.definitions[name] = ("several", [name])
        self._names: dict[str, str] = {}

    def _define(
        self, name: str, opcode: str, operands: list[str], repeated: set[str]
    ) -> None:
        if name in self.definitions:
            repeated.add(name)
        self.definitions[name] = (opcode, operands)

    def name(self, register: str, depth: int = 0) -> str:
        """The register's definition as text, registers replaced by theirs."""
        if register in self._names:
            return self._names[register]
        if register not in self.definitions or depth > 64:
            return register
        opcode, operands = self.definitions[register]
        if opcode == "several":
            return "several"
        text = f"{opcode}({','.join(self._operand(o, depth) for o in operands)})"
        self._names[register] = text
        return text

    def _operand(self, operand: str, depth: int) -> str:
        return _REGISTER.sub(lambda m: self.name(m.group(0), depth + 1), operand)

    def value(self, operand: str, salt: int) -> int:
        """The operand's value: an immediate, or a register's."""
        operand = operand.strip()
        if not operand.startswith("%"):
            return int(operand, 0)
        if operand not in self.definitions:
            return self._free(operand, salt)
        opcode, operands = self.definitions[operand]
        kind = opcode.split(".")[0]
        bits = _width(opcode)
        mask = _MASKS[bits]
        if opcode.startswith("mul.wide."):
            signed = opcode.endswith("s32")
            left, right = (self.value(o, salt) & _MASKS[32] for o in operands)
            if signed:
                left, right = _signed(left, 32), _signed(right, 32)
            return (left * right) & _MASKS[64]
        if opcode in ("cvt.s64.s32", "cvt.u64.u32"):
            (source,) = operands
            value = self.value(source, salt) & _MASKS[32]
            return _signed(value, 32) & _MASKS[64] if "s64" in opcode else value
        if opcode in ("cvt.u32.u64", "cvt.s32.s64"):
            return self.value(operands[0], salt) & _MASKS[32]
        if kind == "mov" and len(operands) == 1 and "{" not in operands[0]:
            source = operands[0]
            if source.startswith("%") and source not in self.definitions:
                return self._free(self.name(operand), salt) & mask
            return self.value(source, salt) & mask
        if kind in ("add", "sub", "shl", "or", "and", "xor") or opcode.startswith(
            "mul.lo."
        ):
            left, right = (self.value(o, salt) for o in operands)
            if kind == "add":
                return (left + right) & mask
            if kind == "sub":
                return (left - right) & mask
            if kind == "shl":
                return (left << right) & mask if right < bits else 0
            if kind == "or":
                return (left | right) & mask
            if kind == "and":
                return left & right & mask
            if kind == "xor":
                return (left ^ right) & mask
            return (left * right) & mask
        return self._free(self.name(operand), salt) & mask

    @staticmethod
    def _free(name: str, salt: int) -> int:
        digest = hashlib.sha256(f"{salt}:{name}".encode()).digest()
        return int.from_bytes(digest[:8], "little")


def store_addresses(
    ptx: str, store: str, salts: tuple[int, ...] = (1, 2, 3)
) -> list[tuple[int, ...]]:
    """The address of each ``store`` instruction of ``ptx`` in program
    order, one number per salt. ``store`` is the instruction with its type,
    such as ``st.global.b16`` or ``st.global.cs.f32``; a plain store's
    ``[register+offset]`` operand and a cache-hinted store inside an inline
    block are both read."""
    kernel = _Kernel(ptx)
    pattern = re.compile(re.escape(store) + r"\s+\[(%\w+)(?:\s*\+\s*(-?\d+))?\]\s*,")
    addresses = []
    for match in pattern.finditer(ptx):
        register, offset = match.group(1), int(match.group(2) or 0)
        addresses.append(
            tuple(
                (kernel.value(register, salt) + offset) & _MASKS[64] for salt in salts
            )
        )
    return addresses
