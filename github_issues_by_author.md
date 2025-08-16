# GitHub Issues Categorized by Author

## Author Statistics
- **Total Issues**: 23
- **Unique Authors**: 12
- **Most Active Author**: v0i0 (4 issues), xmfan (4 issues), pianpwk (4 issues)

## Issues by Author

### **v0i0** (4 issues)
- **#448**: Crash with torch.lerp operation [OPEN]
- **#491**: helion_vmap metaprogramming crashes [OPEN]
- **#492**: Feature request for autotuning stop condition [OPEN]
- **#493**: Arithmetic on input floats incorrectly promotes to fp64 [OPEN]

### **xmfan** (4 issues)
- **#447**: HELION_INTERPRET=1 silently hides HELION_PRINT_OUTPUT_CODE=1 [OPEN]
- **#450**: Crash if @helion.kernel() used without hl.tile [OPEN]
- **#451**: IMA crash with large input sizes (102400x102400) [CLOSED]
- **#454**: Better error message for no autotuning options found [OPEN]

### **pianpwk** (4 issues)
- **#464**: Numerical error with HELION_INTERPRET=1 [OPEN]
- **#466**: HELION_INTERPRET=1 only prints once for fixed block_size [OPEN]
- **#468**: Better error message for rank mismatch in control flow [CLOSED]
- **#496**: "Index out of bounds" error in ref eager mode [OPEN]

### **mlazos** (2 issues)
- **#497**: Tensor size specialization error [OPEN]
- **#498**: Invalid triton kernel generated [CLOSED]

### **StrongerXi** (2 issues)
- **#458**: Type propagation error in pre-processing code [OPEN]
- **#459**: Shape specialization error from reshape [OPEN]

### **anijain2305** (1 issue)
- **#446**: Poor error for tile number mismatch and unhelpful Helion code errors [OPEN]

### **angelayi** (1 issue)
- **#467**: Error when autotuning kernel (missing block size) [OPEN]

### **BoyuanFeng** (1 issue)
- **#452**: Reshape leads to dynamic shape issue [OPEN]

### **exclamaforte** (1 issue)
- **#462**: TypeInferenceError with SymIntType attributes [OPEN]

### **ezyang** (1 issue)
- **#457**: Inductor lowering error missing user stack trace [CLOSED]

### **Lucaskabela** (1 issue)
- **#470**: Out of resources: shared memory error with torch.matmul [OPEN]

### **oulgen** (1 issue)
- **#456**: Limit block sizes in autotuning due to triton max numel range [OPEN]

## Key Observations

### Most Active Contributors
1. **v0i0**: Focus on operator support (lerp), advanced features (vmap), and type system issues
2. **xmfan**: Focus on execution modes (HELION_INTERPRET), kernel structure, and autotuning
3. **pianpwk**: Focus on interpreter mode issues and error messages

### Author Patterns
- **v0i0** tends to report advanced/edge case issues with operators and type system
- **xmfan** and **pianpwk** have significant focus on HELION_INTERPRET mode debugging
- **StrongerXi** has reported multiple shape/type propagation issues
- **mlazos** reported issues with code generation and specialization

### Issue Resolution by Author
- **v0i0**: 0/4 closed (0% resolution rate)
- **xmfan**: 1/4 closed (25% resolution rate)
- **pianpwk**: 1/4 closed (25% resolution rate)
- **mlazos**: 1/2 closed (50% resolution rate)
- **StrongerXi**: 0/2 closed (0% resolution rate)
- **ezyang**: 1/1 closed (100% resolution rate)

### Domain Expertise
- **Interpreter/Debug Mode**: xmfan, pianpwk
- **Type System/Shapes**: StrongerXi, mlazos, exclamaforte
- **Operators/Features**: v0i0
- **Autotuning**: xmfan, angelayi, oulgen
- **Error Messages**: anijain2305, pianpwk, ezyang