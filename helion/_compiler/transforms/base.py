from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ast


class TransformPass:
    def add_kernel_arguments(self, args: list[ast.arg]) -> None:
        """Add any required arguments to the kernel signature.

        Args:
            args: List of kernel arguments to modify
        """

    def add_host_arguments(self, args: list[str]) -> None:
        """Add any required arguments to the host function call.

        Args:
            args: List of host arguments to modify
        """

    def get_host_preamble_statements(self) -> list[ast.AST]:
        """Get statements to inject into the host function preamble.

        Returns:
            List of AST statements to add to the host function
        """
        return []
