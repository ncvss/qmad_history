import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

library_name = "qmad_history"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    use_cuda_if_available = True
    parallelise = True
    vectorise = True
    cuda_error_handling = False

    use_cuda = use_cuda_if_available and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    
    extra_compile_args = {
        "cxx": [
            "-O3",
            #"-g",
        ] + (["-fopenmp"] if parallelise else [])
          + (["-march=native"] if vectorise else []),
        "nvcc": [
            "-O3",
        ],
    }

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))
    if use_cuda:
        sources += cuda_sources


    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            py_limited_api=py_limited_api,
            define_macros = ([("VECTORISATION_ACTIVATED", None)] if vectorise else [])
                            + ([("PARALLELISATION_ACTIVATED", None)] if parallelise else [])
                            + ([("ERROR_HANDLING_OUTPUT", None)] if cuda_error_handling else []),
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)

