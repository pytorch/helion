from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._compiler.source_location import SourceLocation
    from ._compiler.type_propagation import TypeNotAllowedOnDevice


class Base(RuntimeError):
    def report(self) -> str:
        raise NotImplementedError


class _FixedMessage(Base):
    message = ""
    location_suffix = "\nWhile processing:\n{location}"

    def __init__(self, *args: object, **kwargs: object) -> None:
        from ._compiler.source_location import current_location

        self.location: SourceLocation = current_location()
        msg = self.__class__.message.format(*args, **kwargs)
        self.base_msg_len: int = len(msg)
        if self.location and "Original traceback:" not in msg:
            msg += self.location_suffix.format(location=self.location.format())
        super().__init__(msg)


class BaseError(_FixedMessage):
    message = "An error occurred."

    def report(self) -> str:
        return f"ERROR[{type(self).__name__}]: {self!s}"


class NotInsideKernel(BaseError):
    message = (
        "Functions found in helion.language.* must be called from inside a kernel. "
        "Did you forget the @helion.jit decorator?"
    )


class NamingConflict(BaseError):
    message = "The variable name {} is reserved in Helion and cannot be used."


class ClosuresNotSupported(BaseError):
    message = "A closure ({0!r}) was found in the kernel. Closures are not supported."


class ClosureMutation(BaseError):
    message = "Closure mutation (of {0}) is not allowed in a function arg."


class GlobalMutation(BaseError):
    message = "Global mutation (of {0}) is not allowed in a function arg."


class LoopFunctionNotInFor(BaseError):
    message = "{0} must be called from a for loop, e.g. `for ... in {0}(...):"


class InvalidTileUsage(BaseError):
    message = "{0}"


class NestedDeviceLoopsConflict(BaseError):
    message = "Nested device loops must have distinct block sizes."


class DeviceLoopElseBlock(BaseError):
    message = "for...else block is not allowed in a {0} device loop."


class LoopDependencyError(BaseError):
    message = "Loop dependency detected: '{0}' was written in a previous loop."


class TopLevelStatementBetweenLoops(BaseError):
    message = "Statements cannot appear between top level loops."


class NestedGridLoop(BaseError):
    message = "Grid loops must be at the top level of a function."


class RankMismatch(BaseError):
    message = "Expected ndim={expected_ndim}, but got ndim={actual_ndim}{shape_part}. You have {direction}."

    def __init__(
        self, expected_ndim: int, actual_ndim: int, shape_info: str = ""
    ) -> None:
        if actual_ndim > expected_ndim:
            direction = "too many indices"
        elif actual_ndim < expected_ndim:
            direction = "too few indices"
        else:
            direction = "indices that don't match expected structure"

        shape_part = f" ({shape_info})" if shape_info else ""

        super().__init__(
            expected_ndim=expected_ndim,
            actual_ndim=actual_ndim,
            shape_part=shape_part,
            direction=direction,
        )


class InvalidIndexingType(BaseError):
    message = "Expected tile/int/None/tensor/etc in tensor[...], got {0!s}."


class RequiresTensorInAssignment(BaseError):
    message = "Expected tensor in right-hand side of assignment, got {0!s}."


class NotAllowedOnDevice(BaseError):
    message = "The statement {} is not allowed inside the `hl.tile` or `hl.grid` loop."


class ShapeSpecializingCall(BaseError):
    message = "Call would force shape specialization, try `hl.specialize(x)` or `hl.constexpr`."


class ShapeSpecializingAllocation(BaseError):
    message = "Using a tensor size in a device allocation requires specialization. Use `hl.specialize` or `hl.constexpr` to specialize the size."


class SpecializeOnDevice(BaseError):
    message = "hl.specialize() must be called outside the `hl.tile` or `hl.grid` loop."


class SpecializeArgType(BaseError):
    message = "hl.specialize() must be called on a size from an input tensor, got: {}"


class FailedToUnpackTupleAssign(BaseError):
    message = "Failed to unpack values in tuple assignment.  Expected a sequence of size {0}, got type: {1!s}."


class RegisterTunableArgTypes(BaseError):
    message = "expected string literal and ConfigSpecFragment literal, got {0} and {1}"


class TunableTypeNotSupported(BaseError):
    message = "hl.register_tunable() only supports integer, float, and boolean types, got {0!s}."


class TunableNameConflict(BaseError):
    message = (
        "Tunable parameter with name {0!s} already exists. Please use a different name."
    )


class ConfigSpecFragmentWithSymInt(BaseError):
    message = "ConfigSpecFragment with SymInt arg is not supported. hl.constexpr or hl.specialize may be used to specialize the SymInt value."


class FailedToUnpackTile(BaseError):
    message = (
        "Failed to unpack a tile into a tuple assignment. "
        "Expected an sequence, but got a single tile. "
        "Did you mix up `hl.tile(x)` and `hl.tile([x])`?"
    )


class OverpackedTile(BaseError):
    message = (
        "Got a tile wrapped inside a container when indexing a tensor: {0!s}\n"
        "Did you mix up `hl.tile([x])` and `hl.tile(x)`?"
    )


class AssignmentMultipleTargets(NotAllowedOnDevice):
    message = "Assignment with multiple targets (a=b=1) is not allowed inside the `hl.tile` or `hl.grid` loop."


class InvalidAssignment(NotAllowedOnDevice):
    message = "Assignment target must be Name or Subscript inside the `hl.tile` or `hl.grid` loop."


class InvalidSliceType(BaseError):
    message = "Tensor subscript with invalid slice type {0!s}."


class CantReadTypeFromHost(BaseError):
    message = (
        "Loading {0!s} from host is not allowed inside the `hl.tile` or `hl.grid` loop."
    )


class NonTensorSubscriptAssign(BaseError):
    message = "Expected tensor in subscript assignment, got {0!s} and {1!s}."


class ShapeMismatch(BaseError):
    message = "Shape mismatch between {0!s} and {1!s}."


class DeviceAPIOnHost(BaseError):
    message = "{} is only allowed inside the `hl.tile` or `hl.grid` loop."


class StatementNotSupported(BaseError):
    message = "The statement {} is not supported."


class CantReadOnDevice(BaseError):
    message = "Can not read {0!s} inside the `hl.tile` or `hl.grid` loop."


class MaximumGridRank(BaseError):
    message = "Grid can have at most 3 dimensions, got {0}."


class ExpectedTensorName(BaseError):
    message = "Expected tensor name, got {0!s}."


class UndefinedVariable(BaseError):
    message = "{} is not defined."


class StarredArgsNotSupportedOnDevice(BaseError):
    message = "*/** args are not supported inside the `hl.tile` or `hl.grid` loop."


class IncorrectTileUsage(BaseError):
    message = "Tiles can only be used in tensor indexing (`x[tile]`) or in `hl.*` ops (e.g. `hl.zeros(tile)`), used in {}"


class TracedArgNotSupported(BaseError):
    message = "{!s} is not supported as an arg to traced functions."


class NotEnoughConfigs(BaseError):
    message = "FiniteSearch requires at least two configs, but got {0}."


class CantCombineTypesInControlFlow(BaseError):
    message = "Cannot combine types for {0!r} in control flow: {1} and {2}"


class TypePropagationError(BaseError):
    message = "{}"

    def __init__(
        self,
        type_info: TypeNotAllowedOnDevice,
        similar_errors: list[TypePropagationError] | None = None,
    ) -> None:
        from ._compiler.source_location import current_location

        self.locations: list[SourceLocation] = [
            *dict.fromkeys([*type_info.locations, current_location()])
        ]
        if similar_errors is None:
            similar_errors = []
        self.similar_errors: list[TypePropagationError] = similar_errors
        msg = str(type_info)
        self.base_msg_len: int = len(msg)
        msg += self.location_suffix.format(
            location="".join(loc.format() for loc in self.locations)
        )
        super(_FixedMessage, self).__init__(msg)

    @property
    def location(self) -> SourceLocation:
        return self.locations[0]

    def __str__(self) -> str:
        msg = super().__str__()
        if len(self.similar_errors) > 1:
            msg += f"({len(self.similar_errors) - 1} similar errors suppressed)\n"
        return msg


class ErrorCompilingKernel(BaseError):
    message = "{0} errors and {1} warnings occurred (see above)"


class NoTensorArgs(BaseError):
    message = "Kernel took no tensor args, unclear what device to use."


class _WrapException(BaseError):
    message = "{name}: {msg}"

    def __init__(self, e: Exception) -> None:
        super().__init__(name=type(e).__name__, msg=str(e))


class InvalidConfig(BaseError):
    message = "{}"


class InductorLoweringError(BaseError):
    message = "{}"


class DecoratorAfterHelionKernelDecorator(BaseError):
    message = "Decorators after helion kernel decorator are not allowed"


class InternalError(_WrapException):
    pass


class TorchOpTracingError(_WrapException):
    pass


class TritonError(BaseError):
    message = "Error running generated Triton program:\n{1}\n{0}"


class BaseWarning(_FixedMessage):
    message = "A warning occurred."

    def report(self) -> str:
        return f"WARNING[{type(self).__name__}]: {self!s}"


class TensorOperationInWrapper(BaseWarning):
    message = (
        "A tensor operation outside of the `hl.tile` or `hl.grid` loop will not be fused "
        "in the generated kernel.\n"
        "Use @helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper]) to suppress this warning.\n"
        "If this is not a tensor operation, please report this as a bug."
    )


class TensorOperationsInHostCall(TensorOperationInWrapper):
    message = (
        "A tensor operation outside of the `hl.tile` or `hl.grid` loop will not be fused "
        "in the generated kernel: {}"
    )


class WrongDevice(BaseWarning):
    message = "Operation {0} returned a tensor on {1} device, but the kernel is on {2} device. "
