# Issue #462: helion.exc.TypeInferenceError: Attributes are not supported on SymIntType(s77)

## Metadata
- **State**: OPEN
- **Author**: [exclamaforte](https://github.com/exclamaforte)
- **Created**: August 08, 2025 at 00:39 UTC
- **Updated**: August 08, 2025 at 00:42 UTC

## Description

First of all, Helion is some of the most fun I've had writing kernels, great job yall!

**Describe the bug**
Not sure if this is a bug or I'm missing language features, but passing in a normal torch.tensor seems to produce a symint error in my RoPE implementation, which is at the very least surprising to me as a user.

```
Traceback (most recent call last):
  File "/home/gabeferns/helion/benchmarks/run.py", line 583, in <module>
    main()
  File "/home/gabeferns/helion/benchmarks/run.py", line 561, in main
    run_kernel(kernel_names[0], tritonbench_args, input_shard_info)
  File "/home/gabeferns/helion/benchmarks/run.py", line 281, in run_kernel
    run_kernel_variants(
  File "/home/gabeferns/helion/benchmarks/run.py", line 487, in run_kernel_variants
    benchmark.run()
  File "/home/gabeferns/helion/benchmarks/tritonbench/tritonbench/utils/triton_op.py", line 915, in run
    y_vals: Dict[str, BenchmarkOperatorMetrics] = functools.reduce(
  File "/home/gabeferns/helion/benchmarks/tritonbench/tritonbench/utils/triton_op.py", line 903, in _reduce_benchmarks
    acc[bm_name] = self._do_bench(
  File "/home/gabeferns/helion/benchmarks/tritonbench/tritonbench/utils/triton_op.py", line 1204, in _do_bench
    metrics.latency = do_bench_wrapper(
  File "/home/gabeferns/helion/benchmarks/tritonbench/tritonbench/components/do_bench/run.py", line 212, in do_bench_wrapper
    raise e
  File "/home/gabeferns/helion/benchmarks/tritonbench/tritonbench/components/do_bench/run.py", line 202, in do_bench_wrapper
    times=triton.testing.do_bench(
  File "/home/gabeferns/.conda/envs/helion/lib/python3.10/site-packages/triton/testing.py", line 149, in do_bench
    fn()
  File "/home/gabeferns/helion/benchmarks/run.py", line 411, in _inner
    return result()
  File "/home/gabeferns/helion/examples/rope.py", line 212, in <lambda>
    return lambda: rope_kernel(q, k, cos, sin, pos_ids)
  File "/home/gabeferns/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
  File "/home/gabeferns/helion/helion/runtime/kernel.py", line 158, in bind
    bound_kernel = BoundKernel(self, args)
  File "/home/gabeferns/helion/helion/runtime/kernel.py", line 338, in __init__
    self.host_function: HostFunction = HostFunction(
  File "/home/gabeferns/helion/helion/_compiler/host_function.py", line 108, in __init__
    propagate_types(self, fake_args)
  File "/home/gabeferns/helion/helion/_compiler/type_propagation.py", line 2251, in propagate_types
    prop.visit(stmt)
  File "/home/gabeferns/helion/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
  File "/home/gabeferns/helion/helion/_compiler/type_propagation.py", line 1974, in visit_Assign
    type_info = self.visit(node.value)
  File "/home/gabeferns/helion/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
  File "/home/gabeferns/helion/helion/_compiler/type_propagation.py", line 1934, in visit_Attribute
    return value.propagate_attribute(node.attr, origin)
  File "/home/gabeferns/helion/helion/_compiler/type_propagation.py", line 311, in propagate_attribute
    raise exc.TypeInferenceError(f"Attributes are not supported on {self!s}")
helion.exc.TypeInferenceError: Attributes are not supported on SymIntType(s77)
While processing:
  File "/home/gabeferns/helion/examples/rope.py", line 126, in rope_kernel
    b, a2, n, k = k.shape
                  ^^^^^^^
```

**To Reproduce**
https://gist.github.com/exclamaforte/207474539038f4c84732ba2d14952afa


## Comments

### Comment 1 by [exclamaforte](https://github.com/exclamaforte)
*Posted on August 08, 2025 at 00:42 UTC*

`/s/shape/size()/` also shows the same error message

---
