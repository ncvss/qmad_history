# The history of qcd_ml_accel_dirac

This repository serves as a archive of the different optimization steps
done in the qcd_ml_accel_dirac package, so that I can still trace and repeat
those steps later.

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

The vectorisation has the same nomenclature as the loops. If an index is looped and
vectorised, it appears twice.
