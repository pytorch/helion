"""Lowerings helion.inductor puts in Inductor's user_lowerings, which Inductor
consults before its own (one returning None falls back to them)."""

from __future__ import annotations

from .mm import register_mm

__all__ = ["register_mm"]
