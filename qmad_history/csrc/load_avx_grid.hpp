#include <immintrin.h>

namespace qmad_history {

// load function for v and U with the grid layout
inline __m256d load_site (const double * addr){
    // as the values for 2 t sites are next to each other,
    // this is simply a load intrinsic instruction
    return _mm256_loadu_pd(addr);
}

// store function for U and v with grid layout
inline void store_site (double * addr, __m256d a){
    // simply the store intrinsic function
    _mm256_storeu_pd(addr, a);
}


// load function for the hop in negative t direction
inline __m256d load_site_tminus (const double * addr, const double * addrminus){
    return _mm256_loadu2_m128d(addr, addrminus+2);
}

// load function for the hop in positive t direction
inline __m256d load_site_tplus (const double * addr, const double * addrplus){
    return _mm256_loadu2_m128d(addrplus, addr+2);
}

// load function template for any hop, that treats t direction differently
template <int mu, int dir>
inline __m256d load_hop (const double * addr, const double * addrhop){
    // if the hop goes in t direction, we need a specific load function
    if constexpr (mu == 3){
        if constexpr (dir == 0){
            // hop in negative t
            return load_site_tminus(addr, addrhop);
        } else {
            // hop in positive t
            return load_site_tplus(addr, addrhop);
        }
    } else {
        return load_site(addrhop);
    }
}


// the following functions try to optimize the number of accesses,
// but lose performance somewhere else as a result

// // load function for v and U with the grid layout and an output register
// inline void load_site_r (const double * addr, __m256d treg){
//     // as the values for 2 t sites are next to each other,
//     // this is simply a load intrinsic instruction
//     treg = _mm256_loadu_pd(addr);
// }

// // load function for v and U for the neighbor in +t and -t direction at the same time
// // this takes the local t register so that we only need one load from memory
// // we do both registers at the same time so that we only need one permutation
// // addr is the address of the neighboring site
// inline void load_site_tneighbors (const double * addrminus, const double * addrplus,
//                                   __m256d local_reg, __m256d tminus, __m256d tplus){
//     // local_reg // t1r | t1i | t2r | t2i
//     // swap the two complex numbers in the local register
//     __m256d swap_local = _mm256_permute4x64_pd(local_reg, 78); // t2r | t2i | t1r | t1i
//     // the higher part of the local register is the lower part of tplus
//     // the lower part of the local register is the higher part of tminus

//     // the higher part of tplus is the lower site at the t+1 hop address
//     __m128d tplus_hi = _mm_loadu_pd(addrplus); // t3r | t3i
//     // the lower part of tminus is the higher site at the t-1 hop address
//     __m128d tminus_lo = _mm_loadu_pd(addrminus+2); // t0r | t0i

//     tplus = _mm256_insertf64x2(swap_local, tplus_hi, 1); // t2r | t2i | t3r | t3i
//     tminus = _mm256_insertf64x2(swap_local, tminus_lo, 0); // t0r | t0i | t1r | t1i
// }


// // load function for v with the old layout, that swaps the values
// inline __m256d load_split_spin_sw (const double * addr){
//     // low part of the register should be s+1, so the address is increased by 6
//     return _mm256_loadu2_m128d(addr,addr+6);
// }
// // store in v with the old layout
// inline void store_split_spin (double * addr, __m256d a){
//     _mm256_storeu2_m128d(addr+6,addr,a);
// }

}
