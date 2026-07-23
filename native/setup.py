"""Build the optional HEAT native extension without touching the main package."""

from __future__ import annotations

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


debug = os.environ.get("HEAT_NATIVE_DEBUG") == "1"
source_id = os.environ.get("HEAT_NATIVE_SOURCE_ID", "workspace")

if sys.platform == "win32":
    compile_args = ["/std:c++20", "/EHsc"]
    link_args: list[str] = []
    if debug:
        compile_args += ["/Od", "/Zi", "/RTC1"]
        link_args += ["/DEBUG"]
    else:
        compile_args += ["/O2", "/GL", "/fp:precise", "/DNDEBUG"]
        link_args += ["/LTCG"]
else:
    compile_args = ["-std=c++20"]
    link_args = []
    if debug:
        compile_args += ["-O0", "-g"]
    else:
        compile_args += ["-O3", "-fno-fast-math", "-DNDEBUG"]

extensions = [
    Pybind11Extension(
        "heat_native._core",
        ["src/heat_native/core.cpp"],
        cxx_std=20,
        define_macros=[
            ("HEAT_NATIVE_DEBUG", "1" if debug else "0"),
            ("HEAT_NATIVE_SOURCE_ID", f'\"{source_id}\"'),
        ],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(ext_modules=extensions, cmdclass={"build_ext": build_ext})
