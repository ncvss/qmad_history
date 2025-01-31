#include <immintrin.h>

namespace qmad_history {

/**
 * @brief pointwise multiply 2 512 bit registers that each contain 4 complex numbers
 *        in riri format
 * 
 * @param a register that contains 4 complex numbers a1,a2,a3,a4, stored as 8 doubles
 * @param b register that contains 2 complex numbers b1,b2,b3,b4, stored as 8 doubles
 * @return __m512d 
 */
inline __m512d compl_vectorreg_pointwise_mul (__m512d a, __m512d b){

    // a1i | a1r | a2i | a2r | a3i | a3r | a4i | a4r
    __m512d a_re_im_swap = _mm512_permute_pd(a, 85);

    // b1r | b1r | b2r | b2r | b3r | b3r | b4r | b4r
    __m512d b_re = _mm512_permute_pd(b, 0);

    // b1i | b1i | b2i | b2i | b3i | b3i | b4i | b4i
    __m512d b_im = _mm512_permute_pd(b, 255);

    // a1i*b1i | a1r*b1i | a2i*b2i | a2r*b2i | a3i*b3i | a3r*b3i | a4i*b4i | a4r*b4i
    __m512d b_im_summand = _mm512_mul_pd(a_re_im_swap, b_im);

    // a1r*b1r - a1i*b1i | a1i*b1r + a1r*b1i | a2r*b2r - a2i*b2i | a2i*b2r + a2r*b2i |
    // a3r*b3r - a3i*b3i | a3i*b3r + a3r*b3i | a4r*b4r - a4i*b4i | a4i*b4r + a4r*b4i
    return _mm512_fmaddsub_pd(a, b_re, b_im_summand);

}

/**
 * @brief multiply a scalar from a memory pointer onto 4 complex numbers
 *        in a 512 bit register, in riri format
 * 
 * @param bp pointer to a complex number b, stored as 2 doubles (Re b, Im b)
 * @param a register that contains 4 complex numbers a1,a2,a3,a4, stored as 8 doubles
 * @return __m512d 
 */
inline __m512d compl_scalarmem_vectorreg_mul (const double * bp, __m512d a){

    // a1i | a1r | a2i | a2r | a3i | a3r | a4i | a4r
    __m512d a_re_im_swap = _mm512_permute_pd(a, 85);

    // br | br | br | br | br | br | br | br
    __m512d b_re = _mm512_set1_pd(bp[0]);

    // bi | bi | bi | bi | bi | bi | bi | bi
    __m512d b_im = _mm512_set1_pd(bp[1]);
    // note: dereferencing the pointer, then calling set1 becomes a single
    // vbroadcastsd instruction in assembly

    // a1i*bi | a1r*bi | a2i*bi | a2r*bi | a3i*bi | a3r*bi | a4i*bi | a4r*bi
    __m512d b_im_summand = _mm512_mul_pd(a_re_im_swap, b_im);

    // a1r*br - a1i*bi | a1i*br + a1r*bi | a2r*br - a2i*bi | a2i*br + a2r*bi |
    // a3r*br - a3i*bi | a3i*br + a3r*bi | a4r*br - a4i*bi | a4i*br + a4r*bi
    return _mm512_fmaddsub_pd(a, b_re, b_im_summand);

}


/**
 * @brief multiply a complex conjugated scalar from a memory pointer
 *        onto 4 complex numbers in a 512 bit register, in riri format
 * 
 * @param bp pointer to a complex number b, stored as 2 doubles (Re b, Im b)
 * @param a register that contains 4 complex numbers a1,a2,a3,a4, stored as 8 doubles
 * @return __m512d 
 */
inline __m512d compl_scalarmem_conj_vectorreg_mul (const double * bp, __m512d a){

    // a1i | a1r | a2i | a2r | a3i | a3r | a4i | a4r
    __m512d a_re_im_swap = _mm512_permute_pd(a, 85);

    // br | br | br | br | br | br | br | br
    __m512d b_re = _mm512_set1_pd(bp[0]);

    // bi | bi | bi | bi | bi | bi | bi | bi
    __m512d b_im = _mm512_set1_pd(bp[1]);
    // note: dereferencing the pointer, then calling set1 becomes a single
    // vbroadcastsd instruction in assembly

    // a1i*bi | a1r*bi | a2i*bi | a2r*bi | a3i*bi | a3r*bi | a4i*bi | a4r*bi
    __m512d b_im_summand = _mm512_mul_pd(a_re_im_swap, b_im);

    // a1r*br - a1i*bi | a1i*br + a1r*bi | a2r*br - a2i*bi | a2i*br + a2r*bi |
    // a3r*br - a3i*bi | a3i*br + a3r*bi | a4r*br - a4i*bi | a4i*br + a4r*bi
    return _mm512_fmsubadd_pd(a, b_re, b_im_summand);

}



// /**
//  * @brief pointwise multiply 2 512 bit registers that each contain 4 complex numbers,
//  *        with the first number being purely imaginary
//  * 
//  * @param a register that contains 4 purely imaginary numbers a1,a2,a3,a4,
//  *          in the format (a1,a1,a2,a2,a3,a3,a4,a4)
//  * @param b register that contains 4 complex numbers b1,b2,b3,b4, stored as 8 doubles
//  *          in riri format
//  * @return __m256d 
//  */
// inline __m512d imagxcompl_vectorreg_mul (__m512d a, __m512d b){
    
//     // b1i | b1r | b2i | b2r | b3i | b3r | b4i | b4r
//     __m512d b_re_im_swap = _mm512_permute_pd(b, 85);

//     // a1*b1i | a1*b1r | a2*b2i | a2*b2r | a3*b3i | a3*b3r | a4*b4i | a4*b4r
//     __m512d prod = _mm512_mul_pd(a, b_re_im_swap);


//     return _mm512_add

//     __m256d b_swap_reg = _mm256_shuffle_pd(b, b, 5); // b1i | b1r | b2i | b2r
//     __m256d b_mul_reg = _mm256_mul_pd(a, b_swap_reg); // a1i*b1i | a1i*b1r | a2i*b2i | a2i*b2r
//     return _mm256_addsub_pd(_mm256_setzero_pd(), b_mul_reg); // -a1i*b1i | a1i*b1r | -a2i*b2i | a2i*b2r
// }



}
