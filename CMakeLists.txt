cmake_minimum_required(VERSION 3.20)
project(helion_c CXX)

# Helion's C extension layer (see helion/_C/_helion_c.cpp and helion/_C.py).
#
# Built into ``helion._C_ext`` (the actual extension module) and re-exported
# from the pure-Python shim ``helion/_C.py`` as ``helion._C`` so callers
# match the public API (`from helion._C import tensor_key`).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Use scikit-build-core's Python_add_library helper. It creates a shared
# library with the right Python prefix/suffix per platform/Python version
# and links against the appropriate Python runtime.
Python_add_library(_C_ext MODULE WITH_SOABI
  helion/_C/_helion_c.cpp
)

# Disable expensive default warnings; we'll restrict to a small surface area
# and rely on PR review for code quality. Add ``-fvisibility=hidden`` so
# only ``PyInit__C_ext`` is exported.
target_compile_options(_C_ext PRIVATE -O2 -Wall -fvisibility=hidden)

# scikit-build-core installs whatever we tell it to into the wheel.
# Put the .so next to ``helion/_C.py`` so the shim's relative import
# (``from ._C_ext import ...``) resolves.
install(TARGETS _C_ext LIBRARY DESTINATION helion)
