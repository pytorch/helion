# Issue #457: Inductor lowering error no longer has user stack trace

## Metadata
- **State**: CLOSED
- **Author**: [ezyang](https://github.com/ezyang)
- **Created**: August 07, 2025 at 22:40 UTC
- **Updated**: August 13, 2025 at 15:24 UTC
- **Closed**: August 13, 2025 at 15:24 UTC
- **Assignees**: [jansel](https://github.com/jansel)

## Description

https://gist.github.com/ezyang/e6a51fb18f4258c21835c4c3a4e9ab05

fails with

```
Traceback (most recent call last):
  File "/data/users/ezyang/b/helion/examples/roi_align.py", line 137, in <module>
    main()
  File "/data/users/ezyang/b/helion/examples/roi_align.py", line 133, in main
    check()
  File "/data/users/ezyang/b/helion/examples/roi_align.py", line 126, in check
    return roi_align(features, rois, spatial_scale, output_size[0], output_size[1], -1, False)
  File "/data/users/ezyang/b/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
  File "/data/users/ezyang/b/helion/helion/runtime/kernel.py", line 158, in bind
    bound_kernel = BoundKernel(self, args)
  File "/data/users/ezyang/b/helion/helion/runtime/kernel.py", line 338, in __init__
    self.host_function: HostFunction = HostFunction(
  File "/data/users/ezyang/b/helion/helion/_compiler/host_function.py", line 110, in __init__
    self.device_ir = lower_to_device_ir(self)
  File "/data/users/ezyang/b/helion/helion/_compiler/device_ir.py", line 1047, in lower_to_device_ir
    prepare_graph_lowerings(graph.graph)
  File "/data/users/ezyang/b/helion/helion/_compiler/inductor_lowering.py", line 90, in prepare_graph_lowerings
    prepare_node_lowering(graph_lowering, node)
  File "/data/users/ezyang/b/helion/helion/_compiler/inductor_lowering.py", line 164, in prepare_node_lowering
    result = graph_lowering.call_function(
  File "/data/users/ezyang/b/pytorch/torch/_inductor/graph.py", line 1288, in call_function
    raise LoweringException(e, target, args, kwargs).with_traceback(
  File "/data/users/ezyang/b/pytorch/torch/_inductor/graph.py", line 1278, in call_function
    out = lowerings[target](*args, **kwargs)  # type: ignore[index]
  File "/data/users/ezyang/b/pytorch/torch/_inductor/lowering.py", line 458, in wrapped
    out = decomp_fn(*args, **kwargs)
  File "/data/users/ezyang/b/pytorch/torch/_inductor/lowering.py", line 3081, in tensor
    elif len(data) == 0 or isinstance(data[0], (float, int)) and len(data) <= 8:
torch._inductor.exc.LoweringException: TypeError: object of type 'TensorBox' has no len()
  target: aten.scalar_tensor.default
  args[0]: TensorBox(StorageBox(
    InputBuffer(name='scalar_tensor_input0', layout=FixedLayout('cpu', torch.int64, size=[], stride=[]))
  ))
  kwargs: {'dtype': torch.int32, 'layout': torch.strided, 'device': device(type='cpu')}
```

## Comments

*No comments yet.*
