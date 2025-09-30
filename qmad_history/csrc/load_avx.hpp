#include <immintrin.h>

// functions to transfer data between AVX registers and fields with mtsg and tmgs layout

namespace qmad_history {

// load function for v with the old layout
inline __m256d load_split_spin (const double * addr){
    // high part of the register should be s+1, so the address is increased by 6
    return _mm256_loadu2_m128d(addr+6,addr);
}
// load function for v with the old layout, that swaps the values
inline __m256d load_split_spin_sw (const double * addr){
    // low part of the register should be s+1, so the address is increased by 6
    return _mm256_loadu2_m128d(addr,addr+6);
}
// store in v with the old layout
inline void store_split_spin (double * addr, __m256d a){
    _mm256_storeu2_m128d(addr+6,addr,a);
}

}
