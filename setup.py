from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

setup(
    name="helion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Jason Ansel",
    author_email="jansel@meta.com",
    description="A Python-embedded DSL that makes it easy to write ML kernels",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch-labs/helion",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
