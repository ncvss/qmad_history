# Master thesis code

This repository serves as an archive of the code written for my master's thesis.
It contains implementation variants of the Wilson and Wilson clover Dirac
operators with different optimization steps.

## Installation

This package requires Pytorch and Numpy.
The code for the thesis was written specifically for Pytorch 2.4.

For the tests, the packages [GPT](https://github.com/lehner/gpt)
and [qcd_ml](https://github.com/daknuett/qcd_ml) are required.

To install this package, run in the repository directory:
````
pip install . --no-build-isolation
````

In the ``setup.py`` script, some installation options can be adjusted manually:
- ``use_cuda_if_available``: CUDA coda is compiled if the machine is CUDA capable
- ``parallelise``: code is compiled with OpenMP parallelization
- ``vectorise``: code is compiled with AVX vectorization, and functions that use explicit
vectorization are enabled. If disabled, all functions throw an error.
- ``cuda_error_handling``: activates manual error handling for kernel calls in the CUDA code.
This is required to catch some errors that would have no error message at all.
It generates large overhead, so it should not be used when making performance measurements.


## Functionality

The submodules ``wilson`` and `clover` contain classes whose instances represent one
Dirac operator for a specific gauge configuration and choice of parameters $m_0$ and $c_{sw}$.
There is a different class for each mode of initialization of the operator.

Each class has multiple different member functions that each represent one implementation
of the application of the operator to a fermion field. They are called with one fermion field tensor
as the input and ouput the resulting fermion field tensor.

The member function `all_calls` outputs all member functions of the instance that compute the
application of the Dirac operator and are available with the current settings.

The script ``test/test_correct.py`` tests the correctness of all implementations
of the Wilson and Wilson clover Dirac operator against qcd_ml.

## Naming convention

The function names are composed of the following parts:

1. operator type: ``dw`` stand for Dirac Wilson, ``dwc`` for Dirac Wilson Clover
2. code for the way the functions runs
2. memory order of the indices
3. order of the loops over the indices

The indices are named in the following way:
- ``m`` is the hop direction ($\mu$)
- ``d`` is the sign of the hop direction
- ``x`` stands for all space axes
- ``t`` is time
- ``s`` is spin
- ``g`` is gauge
- ``h`` is the gauge that is summed over in the SU(3) matrix multiplication
- ``u`` means a sum is unrolled
- ``n`` is the clover term orientation ($\mu\nu$),
- ``b`` is the blocked space-time
- ``M`` denotes the position of the loop where mass and hop terms
are split
- ``p`` is the parity of the sites (even-odd)

For example, ``dw_dir_mxtsg_xtsgMhm`` means this is a Dirac Wilson operator where there
is no precomputation, the indices from slowest to fastest in memory
are hop direction (``m``), space-time (``t``), spin (``s``), gauge (``g``),
and the loops from outer to inner go over all space directions (``x``), time (``t``),
spin, gauge, gauge that is summed in the matrix multiplication (``h``), hop direction.
The mass term and hop terms are computed in the same gauge and spin loops.

The vectorization has the same nomenclature as the loops. If an index is looped and
vectorized, it appears twice.
There are two different Grid vectorization schemes: for the first, the two nearest
neighbours in t direction are in one register, for the second, the two sites furthest
from each other in t direction are in one register.

## Usage in the thesis

The plots in the thesis were generated from the data ouput by the following scripts:

|figure|script|
|---|---|
|18| `test/thesissteps/wilson_avx_spin_layout_test.py` |
|19 | `test/thesissteps/clover_avx_f_layout_test.py` |
|20| `test/thesissteps/wilson_avx_templ_test.py` |
|21| `test/thesissteps/clover_templ_grid_layout_test.py` |
|22| `test/thesissteps/clover_gradient_impact_test.py` |
|23| `test/thesissteps/wilson_cuda_split_kernel_test.py` |
|24| `test/thesissteps/wilson_cuda_split_sum_test.py` |
|25| `test/thesissteps/clover_cuda_test.py` |
|26| `test/thesissteps/clover_final_comparison.py` |
