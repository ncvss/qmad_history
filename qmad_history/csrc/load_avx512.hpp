#include <immintrin.h>

namespace qmad_history{

static const int spin_addr [] = {0,1,6,7,12,13,18,19};
static const int spin_addr_swap [] = {6,7,0,1,18,19,12,13};

__m512d load_spin (const double* addr){
    return _mm512_i32gather_pd(_mm256_loadu_epi32(spin_addr),addr,1);
}

__m512d load_spin_sw (const double* addr){
    return _mm512_i32gather_pd(_mm256_loadu_epi32(spin_addr_swap),addr,1);
}

// template function to load all 4 spin components into the register
// either after permutation with gamma_mu, or without permutation (if mu is not 0,1,2,3)
template <int mu>
inline __m512d load_spin_gamma (const double* addr){
    if constexpr (mu == 0 || mu == 1){
        return _mm512_i32gather_pd(_mm256_set_epi32(1,0,7,6,13,12,19,18), addr, 1);
    } else {
        if constexpr (mu == 2 || mu == 3){
            return _mm512_i32gather_pd(_mm256_set_epi32(7,6,1,0,19,18,13,12), addr, 1);
        } else {
            return _mm512_i32gather_pd(_mm256_set_epi32(19,18,13,12,7,6,1,0), addr, 1);
        }
    }
}

}