from __future__ import annotations

import glob
import os

# Configuration
EXAMPLES_DIR = "../examples"  # Path to examples directory
RST_DIR = "./examples"  # Directory for individual RST files
EXAMPLES_RST_PATH = "./examples.rst"  # Path to main examples.rst file

# Create the examples directory if it doesn't exist
os.makedirs(RST_DIR, exist_ok=True)

# Get all Python files in the examples directory
example_files = [
    os.path.basename(f) for f in glob.glob(os.path.join(EXAMPLES_DIR, "*.py"))
]
example_files.sort()  # Sort files alphabetically

# Generate individual RST files for each example
for fname in example_files:
    base = os.path.splitext(fname)[0]
    # Capitalize and replace underscores with spaces for nicer titles
    title = base.replace("_", " ").title()
    rst_path = os.path.join(RST_DIR, f"{base}.rst")
    with open(rst_path, "w") as f:
        f.write(
            f"""{title}
{"=" * len(title)}
.. literalinclude:: ../../examples/{fname}
   :language: python
   :linenos:
"""
        )

# Generate the main examples.rst file with toctree
with open(EXAMPLES_RST_PATH, "w") as f:
    f.write(
        """Examples
========

Examples showing the use of Helios in various scenarios.

.. toctree::
   :maxdepth: 1

"""
    )
    # Add each example to the toctree
    for fname in example_files:
        base = os.path.splitext(fname)[0]
        f.write(f"   examples/{base}\n")

print(f"Generated {len(example_files)} example RST files and updated examples.rst")
