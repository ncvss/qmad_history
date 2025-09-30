
namespace qmad_history{
// gamma index for 2 packed spin components
// s=0 is for the spin components 0 and 1, s=2 for th spin components 2 and 3
static const int gamx_pd [3] = {2, 0, 0};

// gamfd[mu][i] is the prefactor of spin component i of gammamu @ v
// complex numbers are stored as 2 doubles
static const double gamfd [32] =
    { 0, 1,   0, 1,   0,-1,   0,-1,
     -1, 0,   1, 0,   1, 0,  -1, 0,
      0, 1,   0,-1,   0,-1,   0, 1,
      1, 0,   1, 0,   1, 0,   1, 0 };
// gamf = [[ i, i,-i,-i],
//         [-1, 1, 1,-1],
//         [ i,-i,-i, i],
//         [ 1, 1, 1, 1] ]

// for 2 spin components in one register, the sigma index is just the spin
// the two numbers have to be swapped for munu=1,2,3,4

// sigf[munu][i] is the prefactor of spin component i of sigmamunu @ v
// complex numbers are stored as 2 doubles
static const double sigfd [48] =
    { 0, 1,   0,-1,   0, 1,   0,-1,
      1, 0,  -1, 0,   1, 0,  -1, 0,
      0, 1,   0, 1,   0, 1,   0, 1,
      0,-1,   0,-1,   0, 1,   0, 1,
      1, 0,  -1, 0,  -1, 0,   1, 0,
      0,-1,   0, 1,   0, 1,   0,-1 };
// sigf = [[ i,-i, i,-i],
//         [ 1,-1, 1,-1],
//         [ i, i, i, i],
//         [-i,-i, i, i],
//         [ 1,-1,-1, 1],
//         [-i, i, i,-i] ]

}
