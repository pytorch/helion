# Summary of Symbolic Shape Fix for codegen_matmul

## Problem
The `codegen_matmul` function in `inductor_lowering.py` was trying to destructure symbolic shapes (SymInt objects) directly:

```python
# This fails when shapes are symbolic (SymInt)
B, H, I, D = lhs_shape
```

When shapes are symbolic, `lhs_shape` is a tuple of `torch.SymInt` objects, not concrete integers. Trying to destructure and use them directly causes issues.

## Solution
The fix properly handles both concrete and symbolic shapes by:

1. **Accessing dimensions by index** instead of destructuring:
   ```python
   B = lhs_shape[0]
   H = lhs_shape[1]
   I = lhs_shape[2]
   D = lhs_shape[3]
   ```

2. **Checking if dimensions are symbolic** and handling them appropriately:
   ```python
   if isinstance(B, torch.SymInt) or isinstance(H, torch.SymInt):
       # Create a symbolic expression for B*H
       B_expr = B._sympy_() if isinstance(B, torch.SymInt) else sympy.sympify(B)
       H_expr = H._sympy_() if isinstance(H, torch.SymInt) else sympy.sympify(H)
       BH_expr = B_expr * H_expr
       BH_str = ctx.cg.device_function.user_sympy_expr(BH_expr)
   else:
       BH_str = str(B * H)
   ```

3. **Converting dimensions to strings** using the appropriate method:
   ```python
   I_str = ctx.cg.device_function.user_sympy_expr(I._sympy_()) if isinstance(I, torch.SymInt) else str(I)
   ```

4. **Using the tile_strategy.shape_str helper** for the final reshape:
   ```python
   out_shape_str = ctx.cg.device_function.tile_strategy.shape_str([B, H, I, J])
   ```

## Key Insights from Other Functions

Looking at other lowering functions in the codebase:
- They use `.size()` to get shapes as tuples
- They access individual dimensions using indexing (e.g., `shape[0]`)
- They use `shape_str()` to convert shape lists to string representations
- They handle SymInts by calling `._sympy_()` to get the underlying sympy expression

## Impact

This fix allows the 4D matmul operation to work with dynamic/symbolic shapes, which is necessary for:
- Dynamic batching scenarios
- Models with variable sequence lengths
- Any case where tensor dimensions aren't known at compile time

The fix maintains backward compatibility with concrete shapes while adding support for symbolic shapes.