#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>


namespace qmad_history{

// lookup tables for the result of the matrix product vmu = gammamu @ v
// gamx[mu][i] is the spin component of v that is proportional to spin component i of vmu
// gamf[mu][i] is the prefactor of spin component i of vmu
// in total, the spin component i of vmu is:
// vmu_{i} = gamf[mu][i] * v_{gamx[mu][i]}

// gamx[mu][i] is the spin component of v that is proportional to spin component i of gammamu @ v
static __constant__ int gamx [4*4] =
    {3, 2, 1, 0,
     3, 2, 1, 0,
     2, 3, 0, 1,
     2, 3, 0, 1 };

// gamf[mu][i] is the prefactor of spin component i of gammamu @ v
static __constant__ c10::complex<double> gamf [4*4] =
    {c10::complex<double>(0, 1), c10::complex<double>(0, 1), c10::complex<double>(0,-1), c10::complex<double>(0,-1),
      -1,   1,   1,  -1,
     c10::complex<double>(0, 1), c10::complex<double>(0,-1), c10::complex<double>(0,-1), c10::complex<double>(0, 1),
       1,   1,   1,   1 };
// gamf = [[ i, i,-i,-i],
//         [-1, 1, 1,-1],
//         [ i,-i,-i, i],
//         [ 1, 1, 1, 1] ]

}
