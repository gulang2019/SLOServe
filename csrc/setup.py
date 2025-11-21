from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "SLOsServe_C",  # Name of the Python module
        ["adm_ctrl_scheduler.cc", "bind.cc", "adm_ctrl_router.cc"],  # Source file
        include_dirs=["."],  # Include the directory containing `adm_ctrl_scheduler.h`
        extra_compile_args=[
            "-O3",              # Max optimization
            "-march=native",    # Enable all CPU-specific optimizations
            "-ffast-math",      # Aggressive floating-point optimizations
            "-funroll-loops",   # Loop unrolling
            "-fno-plt",         # Faster function calls
        ],
    ),
]

setup(
    name="SLOsServe_C",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)