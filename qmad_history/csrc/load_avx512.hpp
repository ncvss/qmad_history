#include <immintrin.h>

// incomplete AVX-512 data transfer functions
// not used in this package or the thesis


namespace qmad_history{

// static const int spin_addr [] = {0,1,6,7,12,13,18,19};
// static const int spin_addr_swap [] = {6,7,0,1,18,19,12,13};

// __m512d load_spin (const double* addr){
//     return _mm512_i32gather_pd(_mm256_loadu_epi32(spin_addr),addr,1);
// }

// __m512d load_spin_sw (const double* addr){
//     return _mm512_i32gather_pd(_mm256_loadu_epi32(spin_addr_swap),addr,1);
// }

// template function to load all 4 spin components into the register
// either after permutation with gamma_mu, or without permutation (if mu is not 0,1,2,3)
template <int mu>
inline __m512d load_spin_g (const double* addr){
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

inline void store_spin (const double* addr, __m512d a){
    _mm512_i32scatter_pd(addr, _mm256_set_epi32(19,18,13,12,7,6,1,0), a, 1);
}


// takes a memory address to the s=0 row of a vector field (with specific g)
// returns a register with all 4 spin components multiplied by gamma_mu
// note: gamma_1,3 is real, so the multiplication is simply a pointwise vector multiplication
//       gamma_0,2 is imaginary, so the real and imaginary entries of a are swapped,
//       then the real entries get a minus sign
template <int mu>
inline __m512d gamma_mul_on_vectormem (const double* addr){
    __m512d a = load_spin_g<mu>(addr);
    if constexpr (mu == 3){
        return a;
    }
    if constexpr (mu == 1){
        return _mm512_mul_pd(a, _mm512_set_pd(-1,-1,  1,1,  1,1,  -1,-1));
    }
    if constexpr (mu == 0){
        __m512d a_re_im_swap = _mm512_permute_pd(a, 85);
        return _mm512_mul_pd(a_re_im_swap, _mm512_set_pd(-1,1,  -1,1,  1,-1,  1,-1));
    }
    if constexpr (mu == 2){
        __m512d a_re_im_swap = _mm512_permute_pd(a, 85);
        return _mm512_mul_pd(a_re_im_swap, _mm512_set_pd(-1,1,  1,-1,  1,-1,  -1,1));
    }

    // if the template input is not 0,1,2,3
    return _mm512_setzero_pd();
}

}