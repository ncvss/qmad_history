#include <immintrin.h>

namespace qmad_history{

static const double gamf_temp [24] =
    { 1, 1,   1, 1,  -1,-1,  -1,-1, // imag
     -1,-1,   1, 1,   1, 1,  -1,-1, // real
      1, 1,  -1,-1,  -1,-1,   1, 1, // imag
      //1, 1,   1, 1,   1, 1,   1, 1 // real
      };

// multiplication with gamf as a template function
template <int M, int S> inline __m256d gamma_mul (__m256d a){
    if constexpr (M == 3){
        return a;
    } else {
        __m256d g_reg = _mm256_loadu_pd(gamf_temp+gixd(M,S));
        if constexpr (M == 0 || M == 2){
            return imagxcompl_vectorreg_mul(g_reg, a);
        } else {
            return _mm256_mul_pd(g_reg,a);
        }
    }
}

static const double sigf_temp [48] = {
      1, 1,  -1,-1,   1, 1,  -1,-1,  // imag
      1, 1,  -1,-1,   1, 1,  -1,-1,  // real
      1, 1,   1, 1,   1, 1,   1, 1,  // imag
     -1,-1,  -1,-1,   1, 1,   1, 1,  // imag
      1, 1,  -1,-1,  -1,-1,   1, 1,  // real
     -1,-1,   1, 1,   1, 1,  -1,-1   // imag
     };

// multiplication with sigf as a template function
template <int MN, int S> inline __m256d sigma_mul (__m256d a){
    __m256d s_reg = _mm256_loadu_pd(sigf_temp+sixd(MN,S));
    if constexpr (MN == 0 || MN == 2 || MN == 3 || MN == 5){
        return imagxcompl_vectorreg_mul(s_reg, a);
    } else {
        return _mm256_mul_pd(s_reg,a);
    }
}

}
