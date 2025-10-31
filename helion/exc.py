from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._compiler.source_location import SourceLocation


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
        "Did you forget the @helion.kernel decorator?"
    )


class NamingConflict(BaseError):
    message = "The variable name {} is reserved in Helion and cannot be used."


class ClosuresNotSupported(BaseError):
    message = "A closure ({0!r}) was found in the kernel. Closures are not supported."


class AutotuneError(BaseError):
    message = "{0}"


class CacheAssertionError(BaseError):
    message = "Expected cache hit for kernel '{0}', but got cache miss. See stderr for diagnostic information."


class ClosureMutation(BaseError):
    message = "Closure mutation (of {0}) is not allowed in a function arg."


class GlobalMutation(BaseError):
    message = "Global mutation (of {0}) is not allowed in a function arg."


class LoopFunctionNotInFor(BaseError):
    message = "{0} must be called from a for loop, e.g. `for ... in {0}(...):"


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


class DotBatchDimensionMismatch(BaseError):
    message = (
        "hl.dot requires matching batch dimensions (or broadcasting with 1s), "
        "but got {lhs} from LHS tensor vs. {rhs} from RHS tensor."
    )


class IndexOffsetOutOfRangeForInt32(BaseError):
    message = (
        "Kernel index_dtype is {0}, but tensor indexing offsets exceed the int32 range. "
        "Use @helion.kernel(index_dtype=torch.int64) to enable larger offsets."
    )


class DataDependentOutputShapeNotSupported(BaseError):
    message = (
        "{op_desc} is not supported in Helion device loops because it produces "
        "a data-dependent output shape."
    )


class UnsupportedSplitOperation(BaseError):
    message = (
        "{op} is not supported in Helion device loops. "
        "For splitting the last dimension with size 2, use hl.split(). "
        "For other splitting operations, consider reshaping or using other hl.* operations."
    )


class RequiresTensorInAssignment(BaseError):
    message = "Expected tensor in right-hand side of assignment, got {0!s}."


class NotAllowedOnDevice(BaseError):
    message = "The statement {} is not allowed inside the `hl.tile` or `hl.grid` loop."


class HostTensorDirectUsage(BaseError):
    message = (
        "Direct use of host tensor '{0}' in op '{1}' not allowed inside the `hl.tile` or `hl.grid` loop. "
        "First load it using {0}[...] or hl.load({0}, ...)."
    )


class ShapeSpecializingCall(BaseError):
    message = "Call would force shape specialization, try `hl.specialize(x)` or `hl.constexpr`."


class ShapeSpecializingAllocation(BaseError):
    message = "Using a tensor size in a device allocation requires specialization. Use `hl.specialize` or `hl.constexpr` to specialize the size."


class SpecializeOnDevice(BaseError):
    message = "hl.specialize() must be called outside the `hl.tile` or `hl.grid` loop."


class SpecializeArgType(BaseError):
    message = "hl.specialize() must be called on a size from an input tensor, got: {}"


class StackTensorcOnHost(BaseError):
    message = "StackTensor must be created inside the `hl.tile` or `hl.grid` loop."


class StackTensorDevPtrOnHost(BaseError):
    message = "StackTensor must be created from a dev_ptr tensor defined on device. Use `hl.load` to load a dev_ptrs tensor. "


class StackTensorDevPtrDtype(BaseError):
    message = (
        "StackTensor must be created from a dev_ptr tensor of dtype int64. Got: {0!s}"
    )


class StackTensorExampleOnDevice(BaseError):
    message = "hl.stacktensor_like must be called with an example host tensor."


class FailedToUnpackTupleAssign(BaseError):
    message = "Failed to unpack values in tuple assignment. Expected a sequence of size {0}, got type: {1!s}."


class RegisterTunableArgTypes(BaseError):
    message = "Expected string literal and ConfigSpecFragment literal, got {0} and {1}."


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
        "Expected a sequence, but got a single tile. "
        "Did you mix up `hl.tile(x)` and `hl.tile([x])`?"
    )


class OverpackedTile(BaseError):
    message = (
        "Got a tile wrapped inside a container when indexing a tensor: {0!s}\n"
        "Did you mix up `hl.tile([x])` and `hl.tile(x)`?"
    )


class InvalidTileRange(BaseError):
    message = (
        "hl.tile() expects the begin of the range to be less than or equal to the end. "
        "Got begin={0!s}, end={1!s}."
    )


class AssignmentMultipleTargets(NotAllowedOnDevice):
    message = "Assignment with multiple targets (a=b=1) is not allowed inside the `hl.tile` or `hl.grid` loop."


class InvalidAssignment(NotAllowedOnDevice):
    message = "Assignment target must be Name or Subscript inside the `hl.tile` or `hl.grid` loop."


class NonTensorSubscriptAssign(BaseError):
    message = "Expected tensor in subscript assignment, got {0!s} and {1!s}."


class ShapeMismatch(BaseError):
    message = "Shape mismatch between {0!s} and {1!s}."


class DeviceAPIOnHost(BaseError):
    message = "{} is only allowed inside the `hl.tile` or `hl.grid` loop."


class StatementNotSupported(BaseError):
    message = "The statement {} is not supported."


class CantReadOnDevice(BaseError):
    message = "Cannot read {0!s} inside the `hl.tile` or `hl.grid` loop."


class BreakpointInDeviceLoopRequiresInterpret(BaseError):
    message = "breakpoint() inside an `hl.tile` or `hl.grid` loop requires TRITON_INTERPRET=1 or HELION_INTERPRET=1."


class UndefinedVariable(BaseError):
    message = "{} is not defined."


class InvalidDeviceForLoop(BaseError):
    message = "For loops on device must use `hl.tile` or `hl.grid`, got {0!s}."


class StarredArgsNotSupportedOnDevice(BaseError):
    message = "*/** args are not supported inside the `hl.tile` or `hl.grid` loop."


class IncorrectTileUsage(BaseError):
    message = "Tiles can only be used in tensor indexing (`x[tile]`) or in `hl.*` ops (e.g. `hl.zeros(tile)`), used in {}"


class TileOfTile(BaseError):
    message = "Expected size arg to `hl.tile` got `Tile`, consider using `hl.tile(other_tile.begin, other_tile.end)`."


class TracedArgNotSupported(BaseError):
    message = "{!s} is not supported as an arg to traced functions."


class NotEnoughConfigs(BaseError):
    message = "FiniteSearch requires at least two configs, but got {0}."


class NoConfigFound(BaseError):
    message = "No working config found from autotuning"


class ReductionOnNonTile(BaseError):
    message = "Reduction must be over a tile or reduction dimension, got {!s}"


class InvalidReductionDim(BaseError):
    message = "Reduction dim must be `int` or `[int]`, got {0!s}."


class MultipleReductionDims(BaseError):
    message = "Multiple reduction dims are not supported."


class ReductionDimInvalidForShape(BaseError):
    message = "Reduction dim {0} is invalid for shape {1!s}."


class CantCombineTypesInControlFlow(BaseError):
    message = "Cannot combine types for {0!r} in control flow: {1} and {2}"


class ErrorCompilingKernel(BaseError):
    message = "{0} errors and {1} warnings occurred (see above)"


class NoTensorArgs(BaseError):
    message = "Kernel took no tensor or device args, unclear what device to use."


class _WrapException(BaseError):
    message = "{name}: {msg}"

    def __init__(self, e: Exception) -> None:
        super().__init__(name=type(e).__name__, msg=str(e))


class InvalidConfig(BaseError):
    message = "{}"


class InductorLoweringError(BaseError):
    message = "{}"


class DecoratorAfterHelionKernelDecorator(BaseError):
    message = "Decorators after helion kernel decorator are not allowed."


class InternalError(_WrapException):
    pass


class TorchOpTracingError(_WrapException):
    pass


class TritonError(BaseError):
    message = """\
Error from Triton code:
{code}

Error running generated Triton program:
{error}
{decorator}
Set autotune_ignore_errors=True or HELION_AUTOTUNE_IGNORE_ERRORS=1 to ignore Triton errors in autotuning."""


class TritonUnrecoverableRuntimeError(BaseError):
    message = """\
An unrecoverable Triton runtime error occurred: {reason}.
This likely indicates a bug in Triton and cannot be recovered from.
{decorator}
Original error: {error}
Set HELION_AUTOTUNE_PRECOMPILE="spawn" to isolate these errors in a subprocess so tuning can continue."""


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
    message = "Operation {0} returned a tensor on {1} device, but the kernel is on {2} device."


class BlockSizeIgnoredInInterpretMode(BaseWarning):
    message = "block_size is specified to be {0}, but in interpret mode, the full dimension size is always used."


class AutotuningDisallowedInEnvironment(BaseError):
    message = "Autotuning is disabled {0}, please provide a config to @helion.kernel via the config= argument."


class UnsupportedPythonType(BaseError):
    message = "{0} is not supported in Helion kernels"


class TypeInferenceError(BaseError):
    message = "{0}"


class ControlFlowTensorMismatch(BaseError):
    message = (
        "Tensor mismatch in control flow for variable '{var}': {details}\n"
        "Hint: ensure the same tensor rank/shape/dtype/device for this variable across branches/iterations."
    )


class NotAllowedInHelperFunction(BaseError):
    message = "This operation is not allowed inside helper functions. It requires kernel context."


class CannotModifyHostVariableOnDevice(BaseError):
    message = "Cannot modify host variable '{0}' inside `hl.tile` or `hl.grid` loop without subscript assignment. Use '{0}[tile] = ...' or '{0}[:] = ...' instead."


class AtomicOnDeviceTensor(BaseError):
    message = (
        "hl.{0}() target must be host-allocated tensor (i.e. allocated outside of hl.tile or hl.grid loop). "
        "Tensors created inside device loops do not have an addressable pointer for atomics."
    )


class CannotReadDeviceVariableOnHost(BaseError):
    message = "Cannot read variable '{0}' defined inside `hl.tile` or `hl.grid` loop from host code."


class DeviceTensorSubscriptAssignmentNotAllowed(BaseError):
    message = "Cannot assign to subscript of device tensor '{0}'."


class InvalidSequenceSubscription(BaseError):
    message = "Cannot subscript a sequence with non constant indices. Got '{0!s}'. "


class InvalidAPIUsage(BaseError):
    message = "Invalid usage of Helion API: {0}"


class GraphModuleUnsupportedOps(BaseError):
    message = "GraphModule contains unsupported operations: {0}. Only pure computation graphs are supported (no load_attr or call_module ops)."


class RefEagerModeCodePrintError(BaseError):
    message = "No generated code to print out if ref eager mode is enabled."


class NoDeviceLoopsInKernel(BaseError):
    message = (
        "Kernel contains no device loops. Add an hl.tile(...) or hl.grid(...) loop "
        "around your device computations."
    )


class NestedKernelCallsNotSupported(BaseError):
    message = (
        "Calling a Helion kernel from within another Helion kernel is not supported. "
        "Helion kernels can only be called from outside of @helion.kernel functions. "
        "If you need to share code between kernels, consider extracting the shared logic "
        "into a regular Python function that can be called from within both kernels."
    )
