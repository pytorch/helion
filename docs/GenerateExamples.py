import os
EXAMPLES_DIR = '../../examples'  # Adjust as needed
RST_DIR = './examples'  # Relative to your Sphinx source dir
example_files = [
    'add.py',
    'all_gather_matmul.py',
    'attention.py',
    'bmm.py',
    'concatenate.py',
    'cross_entropy.py',
    'embedding.py',
    'exp.py',
    'fp8_attention.py',
    'fp8_gemm.py',
    'jagged_dense_add.py',
    'jagged_mean.py',
    'long_sum.py',
    'matmul.py',
    'matmul_layernorm.py',
    'matmul_split_k.py',
    'moe_matmul_ogs.py',
    'rms_norm.py',
    'segment_reduction.py',
    'softmax.py',
    'sum.py',
    'template_via_closure.py',
]
os.makedirs(RST_DIR, exist_ok=True)
for fname in example_files:
    base = os.path.splitext(fname)[0]
    # Capitalize and replace underscores with spaces for nicer titles
    title = base.replace('_', ' ').title()
    rst_path = os.path.join(RST_DIR, f"{base}.rst")
    with open(rst_path, "w") as f:
        f.write(f"""{title}
{'=' * len(title)}
.. literalinclude:: {os.path.join(EXAMPLES_DIR, fname)}
   :language: python
   :linenos:
""")
