from __future__ import annotations

from typing import TYPE_CHECKING

from .. import exc
from .ast_read_writes import ReadWrites
from helion._compat import get_triton_tensor_descriptor_class_import_path
from helion._compat import supports_tensor_descriptor

if TYPE_CHECKING:
    import ast
    from types import FunctionType

SOURCE_MODULE: str = "_source_module"

library_imports: dict[str, str] = {
    "math": "import math",
    "torch": "import torch",
    "helion": "import helion",
    "hl": "import helion.language as hl",
    "triton": "import triton",
    "tl": "import triton.language as tl",
    "triton_helpers": "from torch._inductor.runtime import triton_helpers",
    "tl_math": "from torch._inductor.runtime.triton_helpers import math as tl_math",
    "libdevice": "from torch._inductor.runtime.triton_compat import libdevice",
}

if supports_tensor_descriptor():
    library_imports["TensorDescriptor"] = (
        get_triton_tensor_descriptor_class_import_path()
    )

disallowed_names: dict[str, None] = dict.fromkeys(
    [
        SOURCE_MODULE,
        "make_precompiler",
    ]
)


def get_needed_imports(root: ast.AST) -> str:
    """
    Generate the necessary import statements based on the variables read in the given AST.

    This function analyzes the provided Abstract Syntax Tree (AST) to determine which
    library imports are required based on the variables that are read. It then constructs
    and returns the corresponding import statements.

    :param root: The root AST node to analyze.
    :return: A string containing the required import statements, separated by newlines.
    """
    rw = ReadWrites.from_ast(root)
    result = [library_imports[name] for name in library_imports if name in rw.reads]
    newline = "\n"
    return f"from __future__ import annotations\n\n{newline.join(result)}\n\n"


def assert_no_conflicts(fn: FunctionType) -> None:
    """
    Check for naming conflicts between the function's arguments and reserved names.

    This function verifies that the names used in the provided function do
    not conflict with any reserved names used in the library imports. If
    a conflict is found, an exception is raised.

    :param fn: The function to check for naming conflicts.
    :raises helion.exc.NamingConflict: If a naming conflict is detected.
    """
    for name in fn.__code__.co_varnames:
        if name in library_imports:
            raise exc.NamingConflict(name)
    for name in fn.__code__.co_names:
        if name in library_imports and name in fn.__globals__:
            user_val = fn.__globals__[name]
            scope = {}
            exec(library_imports[name], scope)
            our_val = scope[name]
            if user_val is not our_val:
                raise exc.NamingConflict(name)
        if name in disallowed_names:
            raise exc.NamingConflict(name)
    if fn.__code__.co_freevars:
        raise exc.ClosuresNotSupported(fn.__code__.co_freevars)


def reserved_names() -> list[str]:
    """
    Retrieve a list of reserved names used in the library imports.
    """
    return [*library_imports]
