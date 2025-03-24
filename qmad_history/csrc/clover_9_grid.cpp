// This Dirac Wilson Clover operator uses the memory layout of Grid
// the fastest index goes over neighbouring sites and is as long as the SIMD width
// for me, with AVX, these are two neighboring sites in t direction
// the clover term is also in the grid layout:
// the relevant elements of the tensor product sigma x field strength matrix
// this means we pass the upper triangles of tow 6x6 matrices
// F[x,y,z,t1,triangle index,flattened upper triangle,t2]
// the upper triangle is flattened with the following indices:
//  0 | 1 | 2 | 3 | 4 | 5
//  1 | 6 | 7 | 8 | 9 |10
//  2 | 7 |11 |12 |13 |14
//  3 | 8 |12 |15 |16 |17
//  4 | 9 |13 |16 |18 |19
//  5 |10 |14 |17 |19 |20
// (the lower triangles are the same numbers, but conjugated)


#include <torch/extension.h>

#ifdef VECTORISATION_ACTIVATED

#include <omp.h>
#include <immintrin.h>
#include <iostream>

#include "static/indexfunc_grid.hpp"
#include "complmath_avx.hpp"
#include "load_avx_grid.hpp"


namespace qmad_history{


// gamx[mu][i] is the spin component of v that is proportional to spin component i of gammamu @ v
static const int gamx [4][4] =
    {{3, 2, 1, 0},
     {3, 2, 1, 0},
     {2, 3, 0, 1},
     {2, 3, 0, 1} };


// gamf = [[ i, i,-i,-i],
//         [-1, 1, 1,-1],
//         [ i,-i,-i, i],
//         [ 1, 1, 1, 1] ]

// multiplication of t2 register with gamf as a template function
template <int M, int S> inline __m256d gamma_mul_g (__m256d a){
    if constexpr (M == 0){
        if constexpr (S == 0 || S == 1){
            return compl_reg_times_i(a);
        } else {
            return compl_reg_times_minus_i(a);
        }
    } else {
        if constexpr (M == 1){
            if constexpr (S == 0 || S == 3){
                return _mm256_sub_pd(_mm256_setzero_pd(), a);
            } else {
                return a;
            }
        } else {
            if constexpr (M == 2) {
                if constexpr (S == 0 || S == 3){
                    return compl_reg_times_i(a);
                } else {
                    return compl_reg_times_minus_i(a);
                }
            } else {
                return a;
            }
        }
    }
}


template <int mu, int g, int s>
void dwc_templ_mtsgt_tmgsMht_loop (const double * U, const double * v,
                                   const int * hops, __m256d massf_reg,
                                   double * result, int t, int vol){

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);

    // register with the change in result[t1,s,g]
    __m256d incr;

    if constexpr (mu == 0){
        // mass term is the first term in the result
        incr = load_site(v+vixg(t,g,s));
        incr = _mm256_mul_pd(incr,massf_reg);
    } else {
        // load result of previous computations to add to
        incr = load_site(result+vixg(t,g,s));
    }


    for (int gi = 0; gi < 3; gi++){

        // v hop in negative mu * gammma
        __m256d v_Hmum_gam = load_hop<mu,0>(v+vixg(t,gi,gamx[mu][s]), v+vixg(hops[hixd(t,mu,0)],gi,gamx[mu][s]));
        // multiply the gamma prefactor for
        v_Hmum_gam = gamma_mul_g<mu,s>(v_Hmum_gam);
        
        // v hop in negative mu
        __m256d v_Hmum = load_hop<mu,0>(v+vixg(t,gi,s), v+vixg(hops[hixd(t,mu,0)],gi,s));

        // add those together
        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

        // Umu hop in negative mu, transposed
        __m256d U_Hmum = load_hop<mu,0>(U+uixg(t,mu,gi,g,vol), U+uixg(hops[hixd(t,mu,0)],mu,gi,g,vol));

        // conjugate U and multiply onto v sum
        v_Hmum = compl_vectorreg_conj_pointwise_mul(v_Hmum, U_Hmum);


        // v hop in positive mu * gamma
        __m256d v_Hmup_gam = load_hop<mu,1>(v+vixg(t,gi,gamx[mu][s]), v+vixg(hops[hixd(t,mu,1)],gi,gamx[mu][s]));
        // multiply gamma prefactor
        v_Hmup_gam = gamma_mul_g<mu,s>(v_Hmup_gam);
        
        // v hop in positive mu
        __m256d v_Hmup = load_hop<mu,1>(v+vixg(t,gi,s), v+vixg(hops[hixd(t,mu,1)],gi,s));

        // subtract those 2
        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

        // Umu at this point
        __m256d U_Hmup = load_site(U+uixg(t,mu,g,gi,vol));

        // multiply U onto v sum
        v_Hmup = compl_vectorreg_pointwise_mul(v_Hmup, U_Hmup);

        // add both U*v terms
        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

        // *(-0.5) and add to incr
        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

    }
    // store incr in result
    store_site(result+vixg(t,g,s),incr);
    
}


void dwc_templ_mtsgt_triag_clover (const double * v, const double * F,
                                   double * result, int t){
    
    // upper and lower triangle
    for (int sbl = 0; sbl < 2; sbl++){
        int sbase = sbl*2;

        // 6 registers, each with the wilson term result for one s,g combination
        __m256d r00 = load_site(result+vixg(t,0,sbase+0));
        __m256d r01 = load_site(result+vixg(t,1,sbase+0));
        __m256d r02 = load_site(result+vixg(t,2,sbase+0));
        __m256d r10 = load_site(result+vixg(t,0,sbase+1));
        __m256d r11 = load_site(result+vixg(t,1,sbase+1));
        __m256d r12 = load_site(result+vixg(t,2,sbase+1));
        
        __m256d vreg;
        // v component s=0,g=0
        vreg = load_site(v+vixg(t,0,sbase+0));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,0))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,1))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,2))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,3))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,4))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,5))));

        // v component s=0,g=1
        vreg = load_site(v+vixg(t,1,sbase+0));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,1))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,6))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,7))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,8))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,9))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,10))));

        // v component s=0,g=2
        vreg = load_site(v+vixg(t,2,sbase+0));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,2))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,7))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,11))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,12))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,13))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,14))));

        // v component s=1,g=0
        vreg = load_site(v+vixg(t,0,sbase+1));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,3))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,8))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,12))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,15))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,16))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,17))));

        // v component s=1,g=1
        vreg = load_site(v+vixg(t,1,sbase+1));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,4))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,9))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,13))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,16))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,18))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, load_site(F+fixg(t,sbl,19))));

        // v component s=1,g=2
        vreg = load_site(v+vixg(t,2,sbase+1));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,5))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,10))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,14))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,17))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,19))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_pointwise_mul(vreg, load_site(F+fixg(t,sbl,20))));

        // store into result tensor
        store_site(result+vixg(t,0,sbase+0), r00);
        store_site(result+vixg(t,1,sbase+0), r01);
        store_site(result+vixg(t,2,sbase+0), r02);
        store_site(result+vixg(t,0,sbase+1), r10);
        store_site(result+vixg(t,1,sbase+1), r11);
        store_site(result+vixg(t,2,sbase+1), r12);
    }
    

}


at::Tensor dwc_templ_mtsgt_tmngsMht (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                    const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                                    double mass, double csw){

    // memory layout is U[mu,x,y,z,t1,g,gi,t2] and v[x,y,z,t1,s,gi,t2]
    // t2 are the 2 neighboring sites that are in one register

    TORCH_CHECK(v_tensor.dim() == 7);
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 4);
    TORCH_CHECK(v_tensor.size(5) == 3);
    TORCH_CHECK(v_tensor.size(6) == 2);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4) * 2;
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const double* F = (double*)fs_tensors.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);

    // register for the field strength term prefactor -1/2*csw
    __m256d csw_reg = _mm256_set1_pd(-0.5*csw);

    // vectorization over 2 sites in time
    // thus the space-time loop goes only over half the volume

#pragma omp parallel for
    for (int t1 = 0; t1 < vol/2; t1++){

        // loop over mu=0,1,2,3 g=0,1,2 and s=0,1,2,3 manually with template

        dwc_templ_mtsgt_tmgsMht_loop<0,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,0,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,0,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_tmgsMht_loop<0,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,1,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,1,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_tmgsMht_loop<0,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,2,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<0,2,3>(U,v,hops,massf_reg,result,t1,vol);


        dwc_templ_mtsgt_tmgsMht_loop<1,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,0,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,0,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_tmgsMht_loop<1,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,1,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,1,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_tmgsMht_loop<1,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,2,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<1,2,3>(U,v,hops,massf_reg,result,t1,vol);


        dwc_templ_mtsgt_tmgsMht_loop<2,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,0,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,0,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_tmgsMht_loop<2,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,1,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,1,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_tmgsMht_loop<2,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,2,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<2,2,3>(U,v,hops,massf_reg,result,t1,vol);


        dwc_templ_mtsgt_tmgsMht_loop<3,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,0,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,0,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_tmgsMht_loop<3,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,1,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,1,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_tmgsMht_loop<3,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,2,2>(U,v,hops,massf_reg,result,t1,vol);
        dwc_templ_mtsgt_tmgsMht_loop<3,2,3>(U,v,hops,massf_reg,result,t1,vol);

        dwc_templ_mtsgt_triag_clover(v,F,result,t1);
    }

    return result_tensor;
}

}
#endif
