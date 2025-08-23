from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "promax",  # Name of the Python module
        ["promax_spec_sch.cc", "bind.cc"],  # Source file
        include_dirs=["."],  # Include the directory containing `promax_scheduler.h`
        
    ),
]

setup(
    name="promax",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)