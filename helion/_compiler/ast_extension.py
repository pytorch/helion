from __future__ import annotations

import ast
import enum
import linecache
import os
import re
import textwrap
import threading
import typing
from typing import TYPE_CHECKING
from typing import TypeVar

from .. import exc
from .output_lines import OutputLines
from .source_location import SourceLocation
from .source_location import UnknownLocation
from .source_location import current_location

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .type_propagation import TypeInfo

    _T = TypeVar("_T", bound=ast.AST)
    _R = TypeVar("_R")

    class _TLS(typing.Protocol):
        active_nodes: list[ExtendedAST]


tls: _TLS = typing.cast("_TLS", threading.local())


class LoopType(enum.Enum):
    UNSET = enum.auto()
    HOST = enum.auto()
    GRID = enum.auto()
    DEVICE = enum.auto()


class ExtendedAST:
    """
    We add some extra functionality to the AST classes, by dynamically
    subclassing each AST node class and mixing in this one.
    """

    _fields: tuple[str, ...]

    def __init__(
        self,
        *,
        _location: SourceLocation,
        _type_info: TypeInfo | None = None,
        _loop_type: LoopType = LoopType.UNSET,
        _is_kernel_call: bool = False,
        _root_id: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._type_info: TypeInfo | None = _type_info
        self._location: SourceLocation = _location
        self._loop_type: LoopType = _loop_type
        self._is_kernel_call: bool = _is_kernel_call
        self._root_id: int | None = _root_id

    def new(self, fields: dict[str, object]) -> ExtendedAST:
        result = self.__class__(
            **fields,
            _location=self._location,
            _type_info=self._type_info,
            _loop_type=self._loop_type,
            _is_kernel_call=self._is_kernel_call,
            _root_id=self._root_id,
        )
        return self._location.to_ast(result)

    def fields(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self._fields}

    def copy(self, **changes: object) -> ExtendedAST:
        return self.new({**self.fields(), **changes})

    def __repr__(self) -> str:
        assert isinstance(self, ast.AST)
        return ast.dump(self)

    def update_type_info(self, type_info: TypeInfo) -> TypeInfo:
        if self._type_info is not None and type_info != self._type_info:
            type_info = self._type_info.merge(type_info)
        self._type_info = type_info
        return self._type_info

    def debug_annotations(self) -> list[str]:
        result = []
        if self._type_info:
            result.extend(self._type_info.debug_annotations())
        if self._loop_type != LoopType.UNSET:
            result.append(f"loop_type={self._loop_type.name}")
        return result

    def __enter__(self) -> None:
        try:
            tls.active_nodes.append(self)
        except AttributeError:
            tls.active_nodes = [self]
        self._location.__enter__()

    def __exit__(self, *args: object) -> None:
        self._location.__exit__(*args)
        tls.active_nodes.pop()

    @staticmethod
    def current() -> Sequence[ExtendedAST]:
        """Stack of nodes currently being processed."""
        try:
            return tls.active_nodes
        except AttributeError:
            tls.active_nodes = rv = []
            return rv


_to_extended: dict[type[ast.AST], type[ast.AST]] = {}


def get_wrapper_cls(cls: type[ast.AST]) -> type[ast.AST]:
    if new_cls := _to_extended.get(cls):
        return new_cls

    class Wrapper(ExtendedAST, cls):
        pass

    Wrapper.__name__ = cls.__name__
    rv = typing.cast("type[ast.AST]", Wrapper)
    _to_extended[cls] = rv
    return rv


def create(cls: type[_T], **fields: object) -> _T:
    result = get_wrapper_cls(cls)(**fields, _location=current_location())  # pyright: ignore[reportCallIssue]
    assert isinstance(result, ExtendedAST)
    result._location.to_ast(result)
    return typing.cast("_T", result)


def create_arg(name: str, annotation: str | None = None) -> ast.arg:
    return create(
        ast.arg,
        arg=name,
        annotation=expr_from_string(annotation) if annotation else None,
        type_comment=None,
    )


def create_arguments(args: list[ast.arg]) -> ast.arguments:
    return create(
        ast.arguments,
        args=args,
        posonlyargs=[],
        defaults=[],
        kw_defaults=[],
        kwonlyargs=[],
    )


def statement_from_string(template: str, **placeholders: ast.AST) -> ast.stmt:
    """
    Create an AST statement from a template string with placeholders.

    Uses {placeholder} syntax to mark placeholders that should be replaced with AST nodes.
    This supports two common patterns:

    1. Regular strings - placeholders use single braces:
        expr_from_string("tl.load({ptr} + {offset}, {mask})",
                        ptr=ptr_ast, offset=offset_ast, mask=mask_ast)

    2. f-strings - placeholders use double braces (which become single braces):
        name = "my_tensor"
        expr_from_string(f"tl.load({name} + {{offset}}, {{mask}})",
                        offset=offset_ast, mask=mask_ast)
        # In the f-string, {name} is interpolated to "my_tensor",
        # while {{offset}} becomes {offset} for placeholder replacement
    """
    location: SourceLocation = current_location()

    # Find all placeholders and validate
    pattern = r"\{(\w+)\}(?!:)"  # {word} not followed by colon (avoid dict keys)
    used = set(re.findall(pattern, template))
    if missing := used - placeholders.keys():
        raise KeyError(f"Missing placeholders: {sorted(missing)}")

    # Replace placeholders with unique identifiers to avoid naming conflicts
    # For example, "{x}" in "x = {x}" must not conflict with the variable "x"
    mapping = {}

    def make_unique(m: re.Match[str]) -> str:
        # Extract placeholder name from the regex match (e.g., "offset" from "{offset}")
        name = m.group(1)
        # Create a unique identifier that can't exist in user code
        # Using double underscores and "placeholder" to ensure uniqueness
        uid = f"__placeholder_{len(mapping)}__"
        # Store the mapping from unique ID to the actual AST node
        mapping[uid] = placeholders[name]
        return uid

    # First pass: Replace all {placeholder} with __placeholder_N__ in the template
    # This prevents conflicts and allows ast.parse to create a valid AST
    modified_template = re.sub(pattern, make_unique, template)

    # Parse the modified template into an AST
    (statement,) = ast.parse(modified_template).body

    # Second pass: Recursively walk the AST and replace __placeholder_N__ identifiers
    # with the actual AST nodes provided by the user
    def _replace(node: _R) -> _R:
        # Handle lists by recursively transforming each element
        if isinstance(node, list):
            return [_replace(item) for item in node]  # pyright: ignore[reportReturnType]

        # Pass through non-AST nodes unchanged (e.g., strings, numbers)
        if not isinstance(node, ast.AST):
            return node

        # Replace placeholder names with their corresponding AST nodes
        if isinstance(node, ast.Name) and node.id in mapping:
            return mapping[node.id]  # pyright: ignore[reportReturnType]

        # Recursively transform all child nodes and wrap in ExtendedAST subclass
        cls = get_wrapper_cls(type(node))
        return location.to_ast(  # pyright: ignore[reportReturnType]
            cls(
                **{field: _replace(getattr(node, field)) for field in node._fields},
                _location=location,  # pyright: ignore[reportCallIssue]
            )
        )

    # Apply the second pass transformation to replace all placeholders
    return _replace(statement)


def expr_from_string(template: str, **placeholders: ast.AST) -> ast.AST:
    expr = statement_from_string(template, **placeholders)
    assert isinstance(expr, ast.Expr)
    return expr.value


def convert(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.AST):
        cls = get_wrapper_cls(type(node))
        if "lineno" in node._attributes:
            location = SourceLocation.from_ast(node)
        else:
            # some nodes like arguments lack location information
            location = current_location()
        with location:
            return cls(
                **{field: convert(getattr(node, field)) for field in node._fields},
                **{attr: getattr(node, attr) for attr in node._attributes},
                _location=location,  # pyright: ignore[reportCallIssue]
            )
    elif isinstance(node, list):
        return [convert(item) for item in node]
    else:
        return node


class NodeVisitor(ast.NodeVisitor):
    def visit(self, node: ast.AST) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        with node:
            try:
                visitor = getattr(
                    self,
                    f"visit_{node.__class__.__name__}",
                    self.generic_visit,
                )
                return visitor(node)
            except exc.Base:
                raise
            except Exception as e:
                raise exc.InternalError(e) from e


# Determine whether vanilla ast.unparse keeps parentheses in "(a, b) = c".
# If so, remove the parentheses via `_TupleParensRemovedUnparser` below.
# NOTE: this is to make Python source format consistent between Python 3.10 and 3.12+
_test_src: str = "(a, b) = c"
_needs_to_remove_tuple_parens: bool = (
    ast.unparse(ast.parse(_test_src)).lstrip().startswith("(")
)


class _TupleParensRemovedUnparser(
    ast._Unparser  # pyright: ignore[reportAttributeAccessIssue]
):
    def visit_Tuple(self, node: ast.Tuple) -> None:
        if _needs_to_remove_tuple_parens and isinstance(
            getattr(node, "ctx", None), ast.Store
        ):
            if len(node.elts) == 1:  # single-element tuple
                self.traverse(node.elts[0])
                self.write(",")
            else:  # multi-element tuple
                self.interleave(lambda: self.write(", "), self.traverse, node.elts)
            return
        # For everything else fall back to default behavior
        super().visit_Tuple(node)


class _LocationAnnotatingOutputLines(OutputLines):
    def __init__(self, parent: ast._Unparser) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        super().__init__(parent)
        self._cache: dict[tuple[str, int, int], tuple[str, ...]] = {}
        self._last_location_key: tuple[str, int, int] | None = None

    def reset_last_location(self) -> None:
        super().reset_last_location()
        self._last_location_key = None

    def insert_location_comment(self, location: object) -> None:
        if not isinstance(location, (SourceLocation, UnknownLocation)):
            location = UnknownLocation()
        key = self._location_key(location)
        if key is None or key == self._last_location_key:
            return

        comments = self._comments_for_key(key, location)
        if comments:
            self.insert_comments(comments)
            self._last_location_key = key

    def _location_key(
        self, location: SourceLocation | UnknownLocation
    ) -> tuple[str, int, int] | None:
        if not location:
            return ("<unknown>", 0, 0)
        filename = location.filename
        if not filename:
            return None
        start = location.lineno or 0
        end = location.end_lineno or start
        return (filename, start, end)

    def _comments_for_key(
        self,
        key: tuple[str, int, int],
        location: SourceLocation | UnknownLocation,
    ) -> tuple[str, ...]:
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        filename, start, end = key
        if not location:
            comments = ("# src[unknown]: [source unavailable]",)
        elif start <= 0:
            comments = (
                f"# src[{os.path.basename(filename)}:{start}]: [source unavailable]",
            )
        else:
            lines = linecache.getlines(filename)
            if not lines:
                linecache.checkcache(filename)
                lines = linecache.getlines(filename)

            if not lines:
                comments = (
                    f"# src[{os.path.basename(filename)}:{start}]: [source unavailable]",
                )
            else:
                snippet_full = lines[start - 1 : end]
                if not snippet_full:
                    comments = (
                        f"# src[{os.path.basename(filename)}:{start}]: [source unavailable]",
                    )
                else:
                    max_lines = 3
                    truncated = len(snippet_full) > max_lines
                    snippet = snippet_full[:max_lines]
                    dedented = textwrap.dedent("".join(snippet))
                    body_list: list[str] = []
                    base_name = os.path.basename(filename)
                    for offset, dedented_line in enumerate(dedented.splitlines()):
                        stripped = dedented_line.rstrip()
                        if not stripped.strip():
                            continue
                        lineno = start + offset
                        body_list.append(f"# src[{base_name}:{lineno}]: {stripped}")
                    if truncated:
                        range_part = f"{start}-{end}" if end != start else f"{start}"
                        body_list.append(f"# src[{base_name}:{range_part}]: ...")
                    comments = (
                        tuple(body_list)
                        if body_list
                        else (f"# src[{base_name}:{start}]: [source unavailable]",)
                    )

        self._cache[key] = comments
        return comments


class _HelionUnparser(_TupleParensRemovedUnparser):
    _indent: int

    def __init__(
        self, *args: object, output_origin_lines: bool = True, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        if output_origin_lines:
            self.output = _LocationAnnotatingOutputLines(self)
        else:
            self.output = OutputLines(self)
        self._source = self.output
        self._output_origin_lines = output_origin_lines

    def visit(self, node: ast.AST) -> str:  # type: ignore[override]
        self.output.lines.clear()
        self.output.last_newline = 0
        self.output.reset_last_location()
        self.traverse(node)
        return "".join(self.output)

    def maybe_newline(self) -> None:  # type: ignore[override]
        output = getattr(self, "output", None)
        if output is not None and getattr(output, "_skip_next_newline", False):
            output._skip_next_newline = False
            return
        super().maybe_newline()

    def traverse(self, node: ast.AST | list[ast.AST]) -> None:  # pyright: ignore[reportSignatureIssue]
        if (
            self._output_origin_lines
            and isinstance(node, ExtendedAST)
            and isinstance(node, ast.stmt)
        ):
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                self.output.insert_location_comment(node._location)
        super().traverse(node)


def unparse(ast_obj: ast.AST, *, output_origin_lines: bool = True) -> str:
    unparser = _HelionUnparser(output_origin_lines=output_origin_lines)
    result = unparser.visit(ast_obj)
    del unparser.output  # break reference cycle
    return result
