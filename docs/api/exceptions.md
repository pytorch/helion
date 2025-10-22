# Exceptions

The `helion.exc` module provides an exception hierarchy for error handling and diagnostics.

```{eval-rst}
.. currentmodule:: helion.exc
```

## Overview

Helion's exception system provides detailed error messages with automatic source location tracking. All exceptions inherit from `Base` and provide formatted error reports that help identify exactly where and why compilation failed.

## Exception Hierarchy

### Base Classes

```{eval-rst}
.. autoclass:: Base
   :members:
   :show-inheritance:

.. autoclass:: BaseError
   :members:
   :show-inheritance:

.. autoclass:: BaseWarning
   :members:
   :show-inheritance:
```

## Kernel Context Errors

These exceptions occur when Helion language functions are used incorrectly with respect to kernel context:

```{eval-rst}
.. autoclass:: NotInsideKernel

   Raised when ``helion.language.*`` functions are called outside of a kernel context.

   **Example:**

   .. code-block:: python

      import helion.language as hl

      # This will raise NotInsideKernel
      result = hl.zeros([10])  # Called outside @helion.kernel

.. autoclass:: NotAllowedOnDevice

   Raised when host-only operations are used inside ``hl.tile()`` or ``hl.grid()`` loops.

.. autoclass:: DeviceAPIOnHost

   Raised when device-only APIs are called in host context.

.. autoclass:: CantReadOnDevice

   Raised when attempting to read host variables from device code.

.. autoclass:: NotAllowedInHelperFunction

   Raised when operations requiring kernel context are used in helper functions.
```

## Loop and Control Flow Errors

```{eval-rst}
.. autoclass:: LoopFunctionNotInFor

   Raised when ``hl.tile()`` or ``hl.grid()`` are called outside for loops.

   **Correct usage:**

   .. code-block:: python

      for i in hl.grid(size):  # Correct
          pass

      i = hl.grid(size)        # Raises LoopFunctionNotInFor

.. autoclass:: InvalidDeviceForLoop

   Raised for invalid for loop constructs on device (must use ``hl.tile``/``hl.grid``).

.. autoclass:: NestedDeviceLoopsConflict

   Raised when nested device loops have conflicting block sizes.

.. autoclass:: DeviceLoopElseBlock

   Raised when for...else blocks are used in device loops.

.. autoclass:: NestedGridLoop

   Raised when grid loops are not at function top level.

.. autoclass:: TopLevelStatementBetweenLoops

   Raised when statements appear between top-level loops.

.. autoclass:: LoopDependencyError

   Raised when writing to variables across loop iterations creates dependencies.
```

## Tile and Indexing Errors

```{eval-rst}
.. autoclass:: IncorrectTileUsage

   Raised when tiles are used outside tensor indexing or ``hl.*`` operations.

.. autoclass:: FailedToUnpackTile

   Raised when tuple unpacking fails for single tile.

.. autoclass:: InvalidTileRange

   Raised when ``hl.tile`` is given a range where the begin exceeds the end.

.. autoclass:: OverpackedTile

   Raised when tile is wrapped in container when indexing.

.. autoclass:: RankMismatch

   Raised when tensor rank doesn't match indexing dimensions.

   **Example:**

   .. code-block:: python

      x = torch.randn(10, 20)  # 2D tensor
      for i in hl.grid(10):
          y = x[i, j, k]       # Raises RankMismatch - too many indices

.. autoclass:: InvalidIndexingType

   Raised for invalid types in tensor subscripts.

.. autoclass:: DataDependentOutputShapeNotSupported

   Raised when operations that produce data-dependent output shapes are used inside Helion device loops.

.. autoclass:: HostTensorDirectUsage

    Raised when host tensors are used directly in device code without proper indexing.
```

## Assignment and Variable Errors

```{eval-rst}
.. autoclass:: RequiresTensorInAssignment

   Raised when non-tensor appears on RHS of assignment.

.. autoclass:: NonTensorSubscriptAssign

   Raised for invalid types in subscript assignment.

.. autoclass:: AssignmentMultipleTargets

   Raised for multiple assignment targets (``a=b=1``) on device.

.. autoclass:: InvalidAssignment

   Raised for invalid assignment targets on device.

.. autoclass:: FailedToUnpackTupleAssign

   Raised when tuple unpacking fails in assignment.

.. autoclass:: ShapeMismatch

   Raised for shape incompatibility between tensors.

.. autoclass:: UndefinedVariable

   Raised when referencing undefined variables.

.. autoclass:: CannotModifyHostVariableOnDevice

   Raised when modifying host variables inside device loops without subscript assignment.

.. autoclass:: CannotReadDeviceVariableOnHost

   Raised when attempting to read variables defined inside device loops from host context.

.. autoclass:: DeviceTensorSubscriptAssignmentNotAllowed

   Raised when attempting to assign to subscript of device tensor.
```

## Type and Inference Errors

```{eval-rst}
.. autoclass:: UnsupportedPythonType

   Raised for Python types not supported in kernels.

.. autoclass:: TypeInferenceError

   Raised for type inference failures.

.. autoclass:: CantCombineTypesInControlFlow

   Raised for type conflicts in control flow.

.. autoclass:: TracedArgNotSupported

   Raised for unsupported argument types in traced functions.

.. autoclass:: InvalidAPIUsage

   Raised for incorrect usage of Helion API functions.
```

## Configuration Errors

```{eval-rst}
.. autoclass:: InvalidConfig

   Raised for invalid kernel configurations.

.. autoclass:: NotEnoughConfigs

   Raised when insufficient configs provided for FiniteSearch.

.. autoclass:: ShapeSpecializingCall

   Raised for calls requiring shape specialization.

.. autoclass:: ShapeSpecializingAllocation

   Raised for allocations requiring specialization.

.. autoclass:: SpecializeOnDevice

   Raised when ``hl.specialize()`` is called in device loop.

.. autoclass:: SpecializeArgType

   Raised for invalid arguments to ``hl.specialize()``.

.. autoclass:: ConfigSpecFragmentWithSymInt

   Raised for ConfigSpecFragment with SymInt.

.. autoclass:: AutotuningDisallowedInEnvironment

   Raised when autotuning is disabled in environment and no config is provided.
```

## Tunable Parameter Errors

```{eval-rst}
.. autoclass:: RegisterTunableArgTypes

   Raised for invalid argument types to ``hl.register_tunable()``.

.. autoclass:: TunableTypeNotSupported

   Raised for unsupported tunable parameter types.

.. autoclass:: TunableNameConflict

   Raised for duplicate tunable parameter names.
```

## Language and Syntax Errors

```{eval-rst}
.. autoclass:: NamingConflict

   Raised when reserved variable names are used.

.. autoclass:: ClosuresNotSupported

   Raised when closures are found in kernels.

.. autoclass:: ClosureMutation

   Raised for closure variable mutation.

.. autoclass:: GlobalMutation

   Raised for global variable mutation.

.. autoclass:: StatementNotSupported

   Raised for unsupported statement types.

.. autoclass:: StarredArgsNotSupportedOnDevice

   Raised for ``*``/``**`` args in device loops.

.. autoclass:: DecoratorAfterHelionKernelDecorator

   Raised when decorators appear after ``@helion.kernel``.
```

## Grid and Execution Errors

```{eval-rst}
.. autoclass:: NoTensorArgs

   Raised for kernels with no tensor arguments.

```

## Compilation and Runtime Errors

```{eval-rst}
.. autoclass:: ErrorCompilingKernel

   Raised for compilation failures with error/warning counts.

.. autoclass:: TritonError

   Raised for errors in generated Triton programs.

.. autoclass:: InductorLoweringError

   Raised for Inductor lowering failures.

.. autoclass:: TorchOpTracingError

   Raised for Torch operation tracing errors.

.. autoclass:: InternalError

   Raised for internal compiler errors.
```

## Warning Classes

Warnings can be suppressed by including them in the `ignore_warnings` setting:

```{eval-rst}
.. autoclass:: TensorOperationInWrapper

   Warns when tensor operations occur outside ``hl.tile``/``hl.grid`` loops.

.. autoclass:: TensorOperationsInHostCall

   Specific variant for tensor operations in host calls.

.. autoclass:: WrongDevice

   Warns when operations return tensors on wrong device.

```

### Warning Suppression

```python
# Suppress specific warnings
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def my_kernel(x):
    y = x * 2  # This operation won't be fused, but warning suppressed
    for i in hl.grid(x.size(0)):
        pass


# Suppress ALL warnings by using BaseWarning
@helion.kernel(ignore_warnings=[helion.exc.BaseWarning])
def quiet_kernel(x):
    # This kernel will suppress all Helion warnings
    pass
```

## See Also

- {doc}`settings` - Configuring debug output and warning suppression
- {doc}`kernel` - Kernel execution and error contexts
