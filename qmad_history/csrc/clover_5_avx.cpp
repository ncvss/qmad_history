// Dirac Wilson Clover using avx vectorization of the spin

#include <torch/extension.h>

#ifdef VECTORISATION_ACTIVATED

#include <omp.h>
#include <immintrin.h>

#include "static/indexfunc_double.hpp"
#include "static/gamma_double.hpp"
#include "complmath_avx.hpp"
#include "load_avx.hpp"

namespace qmad_history{

// load function for v for 2 spins with difference 2
inline __m256d load_split2_spin (const double * addr){
    // high part of the register should be s+2, so the address is increased by 12
    return _mm256_loadu2_m128d(addr+12,addr);
}
// store in v for 2 spins with difference 2
inline void store_split2_spin (double * addr, __m256d a){
    _mm256_storeu2_m128d(addr+12,addr,a);
}
// index function for F in layout F[x,y,z,t,flattened upper triangle,block index]
inline int fixo (int t, int triix){
    return t*84 + triix*4;
}

at::Tensor dwc_avx_tmgs_tmgsMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_tensor, double mass, double csw){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(0) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 3);
    TORCH_CHECK(v_tensor.size(5) == 4);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(0) * U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3);
    

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

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);

    // register for the field strength term prefactor -1/2*csw
    __m256d csw_reg = _mm256_set1_pd(-0.5*csw);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // mu loop partially unrolled
        // for (int mu = 0; mu < 4; mu++){
        int mu = 0;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // mass term is the first term in the result
                __m256d incr = _mm256_loadu_pd(v+vixd(t,g,s));
                incr = _mm256_mul_pd(incr,massf_reg);

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmum_gam = _mm256_permute4x64_pd(v_Hmum_gam,78);
                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmup_gam = _mm256_permute4x64_pd(v_Hmup_gam,78);
                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);


                    // now the field strength tensor for the clover improvement
                    
                    // v at point t, without any swap in s (for munu=0,5)
                    __m256d v_F_05 = _mm256_loadu_pd(v+vixd(t,gi,s));
                    // v at point t, with the 2 numbers in the register swapped (for munu=1,2,3,4)
                    __m256d v_F_1234 = _mm256_permute4x64_pd(v_F_05,78);

                    // sigma prefactor for munu=0
                    __m256d sigf = _mm256_loadu_pd(sigfd+sixd(0,s));
                    // improvement for munu=0
                    __m256d v_F_0 = compl_vectorreg_pointwise_mul(sigf,v_F_05);
                    v_F_0 = compl_scalarmem_vectorreg_mul(F+fixd(t,0,g,gi),v_F_0);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_0,csw_reg,incr);

                    // sigma prefactor for munu=1
                    sigf = _mm256_loadu_pd(sigfd+sixd(1,s));
                    // improvement for munu=1
                    __m256d v_F_1 = compl_vectorreg_pointwise_mul(sigf,v_F_1234);
                    v_F_1 = compl_scalarmem_vectorreg_mul(F+fixd(t,1,g,gi),v_F_1);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_1,csw_reg,incr);

                    // sigma prefactor for munu=2
                    sigf = _mm256_loadu_pd(sigfd+sixd(2,s));
                    // improvement for munu=2
                    __m256d v_F_2 = compl_vectorreg_pointwise_mul(sigf,v_F_1234);
                    v_F_2 = compl_scalarmem_vectorreg_mul(F+fixd(t,2,g,gi),v_F_2);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_2,csw_reg,incr);

                    // sigma prefactor for munu=3
                    sigf = _mm256_loadu_pd(sigfd+sixd(3,s));
                    // improvement for munu=3
                    __m256d v_F_3 = compl_vectorreg_pointwise_mul(sigf,v_F_1234);
                    v_F_3 = compl_scalarmem_vectorreg_mul(F+fixd(t,3,g,gi),v_F_3);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_3,csw_reg,incr);

                    // sigma prefactor for munu=4
                    sigf = _mm256_loadu_pd(sigfd+sixd(4,s));
                    // improvement for munu=4
                    __m256d v_F_4 = compl_vectorreg_pointwise_mul(sigf,v_F_1234);
                    v_F_4 = compl_scalarmem_vectorreg_mul(F+fixd(t,4,g,gi),v_F_4);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_4,csw_reg,incr);

                    // sigma prefactor for munu=5
                    sigf = _mm256_loadu_pd(sigfd+sixd(5,s));
                    // improvement for munu=5
                    __m256d v_F_5 = compl_vectorreg_pointwise_mul(sigf,v_F_05);
                    v_F_5 = compl_scalarmem_vectorreg_mul(F+fixd(t,5,g,gi),v_F_5);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_5,csw_reg,incr);

                }
                // store incr in result
                _mm256_storeu_pd(result+vixd(t,g,s),incr);
            }
        }
        mu = 1;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // load result of previous computations to add to
                __m256d incr = _mm256_loadu_pd(result+vixd(t,g,s));

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmum_gam = _mm256_permute4x64_pd(v_Hmum_gam,78);
                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmup_gam = _mm256_permute4x64_pd(v_Hmup_gam,78);
                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                _mm256_storeu_pd(result+vixd(t,g,s),incr);
            }
        }
        for (mu = 2; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s+=2){

                    // load result of previous computations to add to
                    __m256d incr = _mm256_loadu_pd(result+vixd(t,g,s));

                    // gamma_mu prefactor for the spins s and s+1
                    __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                    for (int gi = 0; gi < 3; gi++){
                        // v hop in negative mu * gammma
                        __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                        // multiply the gamma prefactor for s and s+1
                        v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                        // v hop in negative mu
                        __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                        // add those together
                        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                        // v hop in positive mu * gamma
                        __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                        v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                        // v hop in positive mu
                        __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                        // subtract those 2
                        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                        // multiply U at this point onto v sum
                        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                        // add both U*v terms
                        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                        // *(-0.5) and add to incr
                        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                    }
                    // store incr in result
                    _mm256_storeu_pd(result+vixd(t,g,s),incr);
                }
            }
        }
        //} end mu loop

    }

    return result_tensor;
}


at::Tensor dwc_avx_mtsg_tmgsMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_tensor, double mass, double csw){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 4);
    TORCH_CHECK(v_tensor.size(5) == 3);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

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

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);

    // register for the field strength term prefactor -1/2*csw
    __m256d csw_reg = _mm256_set1_pd(-0.5*csw);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // mu loop partially unrolled
        // for (int mu = 0; mu < 4; mu++){
        int mu = 0;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // mass term is the first term in the result
                __m256d incr = load_split_spin(v+vixo(t,g,s));
                incr = _mm256_mul_pd(incr,massf_reg);

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                // sigma_munu prefactors for all munu for s and s+1
                __m256d sigf0 = _mm256_loadu_pd(sigfd+sixd(0,s));
                __m256d sigf1 = _mm256_loadu_pd(sigfd+sixd(1,s));
                __m256d sigf2 = _mm256_loadu_pd(sigfd+sixd(2,s));
                __m256d sigf3 = _mm256_loadu_pd(sigfd+sixd(3,s));
                __m256d sigf4 = _mm256_loadu_pd(sigfd+sixd(4,s));
                __m256d sigf5 = _mm256_loadu_pd(sigfd+sixd(5,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);


                    // now the field strength tensor for the clover improvement
                    
                    // v at point t, without any swap in s (for munu=0,5)
                    __m256d v_F_05 = load_split_spin(v+vixo(t,gi,s));
                    // v at point t, with the 2 numbers in the register swapped (for munu=1,2,3,4)
                    __m256d v_F_1234 = _mm256_permute4x64_pd(v_F_05,78);

                    // improvement for munu=0
                    __m256d v_F_0 = compl_vectorreg_pointwise_mul(sigf0,v_F_05);
                    v_F_0 = compl_scalarmem_vectorreg_mul(F+fixd(t,0,g,gi),v_F_0);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_0,csw_reg,incr);

                    // improvement for munu=1
                    __m256d v_F_1 = compl_vectorreg_pointwise_mul(sigf1,v_F_1234);
                    v_F_1 = compl_scalarmem_vectorreg_mul(F+fixd(t,1,g,gi),v_F_1);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_1,csw_reg,incr);

                    // improvement for munu=2
                    __m256d v_F_2 = compl_vectorreg_pointwise_mul(sigf2,v_F_1234);
                    v_F_2 = compl_scalarmem_vectorreg_mul(F+fixd(t,2,g,gi),v_F_2);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_2,csw_reg,incr);

                    // improvement for munu=3
                    __m256d v_F_3 = compl_vectorreg_pointwise_mul(sigf3,v_F_1234);
                    v_F_3 = compl_scalarmem_vectorreg_mul(F+fixd(t,3,g,gi),v_F_3);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_3,csw_reg,incr);

                    // improvement for munu=4
                    __m256d v_F_4 = compl_vectorreg_pointwise_mul(sigf4,v_F_1234);
                    v_F_4 = compl_scalarmem_vectorreg_mul(F+fixd(t,4,g,gi),v_F_4);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_4,csw_reg,incr);

                    // improvement for munu=5
                    __m256d v_F_5 = compl_vectorreg_pointwise_mul(sigf5,v_F_05);
                    v_F_5 = compl_scalarmem_vectorreg_mul(F+fixd(t,5,g,gi),v_F_5);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_5,csw_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        mu = 1;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // load result of previous computations to add to
                __m256d incr = load_split_spin(result+vixo(t,g,s));

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        for (mu = 2; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s+=2){

                    // load result of previous computations to add to
                    __m256d incr = load_split_spin(result+vixo(t,g,s));

                    // gamma_mu prefactor for the spins s and s+1
                    __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                    for (int gi = 0; gi < 3; gi++){
                        // v hop in negative mu * gammma
                        __m256d v_Hmum_gam = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                        // multiply the gamma prefactor for s and s+1
                        v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                        // v hop in negative mu
                        __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                        // add those together
                        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                        // v hop in positive mu * gamma
                        __m256d v_Hmup_gam = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                        v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                        // v hop in positive mu
                        __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                        // subtract those 2
                        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                        // multiply U at this point onto v sum
                        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                        // add both U*v terms
                        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                        // *(-0.5) and add to incr
                        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                    }
                    // store incr in result
                    store_split_spin(result+vixo(t,g,s),incr);
                }
            }
        }
        //} end mu loop

    }

    return result_tensor;
}


// as I chose msg to be the best order, I have to add it here as well

at::Tensor dwc_avx_mtsg_tmsgMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_tensor, double mass, double csw){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 4);
    TORCH_CHECK(v_tensor.size(5) == 3);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

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

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);

    // register for the field strength term prefactor -1/2*csw
    __m256d csw_reg = _mm256_set1_pd(-0.5*csw);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // mu loop partially unrolled
        // for (int mu = 0; mu < 4; mu++){
        int mu = 0;
        for (int s = 0; s < 4; s+=2){
            for (int g = 0; g < 3; g++){
            

                // mass term is the first term in the result
                __m256d incr = load_split_spin(v+vixo(t,g,s));
                incr = _mm256_mul_pd(incr,massf_reg);

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                // sigma_munu prefactors for all munu for s and s+1
                __m256d sigf0 = _mm256_loadu_pd(sigfd+sixd(0,s));
                __m256d sigf1 = _mm256_loadu_pd(sigfd+sixd(1,s));
                __m256d sigf2 = _mm256_loadu_pd(sigfd+sixd(2,s));
                __m256d sigf3 = _mm256_loadu_pd(sigfd+sixd(3,s));
                __m256d sigf4 = _mm256_loadu_pd(sigfd+sixd(4,s));
                __m256d sigf5 = _mm256_loadu_pd(sigfd+sixd(5,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);


                    // now the field strength tensor for the clover improvement
                    
                    // v at point t, without any swap in s (for munu=0,5)
                    __m256d v_F_05 = load_split_spin(v+vixo(t,gi,s));
                    // v at point t, with the 2 numbers in the register swapped (for munu=1,2,3,4)
                    __m256d v_F_1234 = _mm256_permute4x64_pd(v_F_05,78);

                    // improvement for munu=0
                    __m256d v_F_0 = compl_vectorreg_pointwise_mul(sigf0,v_F_05);
                    v_F_0 = compl_scalarmem_vectorreg_mul(F+fixd(t,0,g,gi),v_F_0);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_0,csw_reg,incr);

                    // improvement for munu=1
                    __m256d v_F_1 = compl_vectorreg_pointwise_mul(sigf1,v_F_1234);
                    v_F_1 = compl_scalarmem_vectorreg_mul(F+fixd(t,1,g,gi),v_F_1);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_1,csw_reg,incr);

                    // improvement for munu=2
                    __m256d v_F_2 = compl_vectorreg_pointwise_mul(sigf2,v_F_1234);
                    v_F_2 = compl_scalarmem_vectorreg_mul(F+fixd(t,2,g,gi),v_F_2);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_2,csw_reg,incr);

                    // improvement for munu=3
                    __m256d v_F_3 = compl_vectorreg_pointwise_mul(sigf3,v_F_1234);
                    v_F_3 = compl_scalarmem_vectorreg_mul(F+fixd(t,3,g,gi),v_F_3);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_3,csw_reg,incr);

                    // improvement for munu=4
                    __m256d v_F_4 = compl_vectorreg_pointwise_mul(sigf4,v_F_1234);
                    v_F_4 = compl_scalarmem_vectorreg_mul(F+fixd(t,4,g,gi),v_F_4);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_4,csw_reg,incr);

                    // improvement for munu=5
                    __m256d v_F_5 = compl_vectorreg_pointwise_mul(sigf5,v_F_05);
                    v_F_5 = compl_scalarmem_vectorreg_mul(F+fixd(t,5,g,gi),v_F_5);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_5,csw_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        mu = 1;
        for (int s = 0; s < 4; s+=2){
            for (int g = 0; g < 3; g++){

                // load result of previous computations to add to
                __m256d incr = load_split_spin(result+vixo(t,g,s));

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        for (mu = 2; mu < 4; mu++){
            for (int s = 0; s < 4; s+=2){
                for (int g = 0; g < 3; g++){

                    // load result of previous computations to add to
                    __m256d incr = load_split_spin(result+vixo(t,g,s));

                    // gamma_mu prefactor for the spins s and s+1
                    __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                    for (int gi = 0; gi < 3; gi++){
                        // v hop in negative mu * gammma
                        __m256d v_Hmum_gam = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                        // multiply the gamma prefactor for s and s+1
                        v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                        // v hop in negative mu
                        __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                        // add those together
                        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                        // v hop in positive mu * gamma
                        __m256d v_Hmup_gam = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                        v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                        // v hop in positive mu
                        __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                        // subtract those 2
                        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                        // multiply U at this point onto v sum
                        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                        // add both U*v terms
                        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                        // *(-0.5) and add to incr
                        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                    }
                    // store incr in result
                    store_split_spin(result+vixo(t,g,s),incr);
                }
            }
        }
        //} end mu loop

    }

    return result_tensor;
}


// non-template version of the grid layout of sigma field strength

at::Tensor dwc_avx_mtsg_tmnsgMhs_sigpre (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_tensor, double mass){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 4);
    TORCH_CHECK(v_tensor.size(5) == 3);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

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

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // mu loop partially unrolled
        // for (int mu = 0; mu < 4; mu++){
        int mu = 0;
        for (int s = 0; s < 4; s+=2){
            for (int g = 0; g < 3; g++){
            

                // mass term is the first term in the result
                __m256d incr = load_split_spin(v+vixo(t,g,s));
                incr = _mm256_mul_pd(incr,massf_reg);

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));


                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);


                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        mu = 1;
        for (int s = 0; s < 4; s+=2){
            for (int g = 0; g < 3; g++){

                // load result of previous computations to add to
                __m256d incr = load_split_spin(result+vixo(t,g,s));

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        for (mu = 2; mu < 4; mu++){
            for (int s = 0; s < 4; s+=2){
                for (int g = 0; g < 3; g++){

                    // load result of previous computations to add to
                    __m256d incr = load_split_spin(result+vixo(t,g,s));

                    // gamma_mu prefactor for the spins s and s+1
                    __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                    for (int gi = 0; gi < 3; gi++){
                        // v hop in negative mu * gammma
                        __m256d v_Hmum_gam = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                        // multiply the gamma prefactor for s and s+1
                        v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                        // v hop in negative mu
                        __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                        // add those together
                        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                        // v hop in positive mu * gamma
                        __m256d v_Hmup_gam = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                        v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                        // v hop in positive mu
                        __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                        // subtract those 2
                        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                        // multiply U at this point onto v sum
                        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                        // add both U*v terms
                        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                        // *(-0.5) and add to incr
                        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                    }
                    // store incr in result
                    store_split_spin(result+vixo(t,g,s),incr);
                }
            }
        }
        //} end mu loop

        // now the field strength sigma product tensor for the clover improvement
                    
        // 6 registers, each with the wilson term result for one s,g combination
        // the vectorization is over s, one register has s=0,2 or s=1,3
        __m256d r00 = load_split2_spin(result+vixo(t,0,0));
        __m256d r01 = load_split2_spin(result+vixo(t,1,0));
        __m256d r02 = load_split2_spin(result+vixo(t,2,0));
        __m256d r10 = load_split2_spin(result+vixo(t,0,1));
        __m256d r11 = load_split2_spin(result+vixo(t,1,1));
        __m256d r12 = load_split2_spin(result+vixo(t,2,1));
        
        __m256d vreg;
        // v component s=0,g=0
        vreg = load_split2_spin(v+vixo(t,0,0));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,0))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,1))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,2))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,3))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,4))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,5))));

        // v component s=0,g=1
        vreg = load_split2_spin(v+vixo(t,1,0));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,1))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,6))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,7))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,8))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,9))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,10))));

        // v component s=0,g=2
        vreg = load_split2_spin(v+vixo(t,2,0));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,2))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,7))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,11))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,12))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,13))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,14))));

        // v component s=1,g=0
        vreg = load_split2_spin(v+vixo(t,0,1));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,3))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,8))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,12))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,15))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,16))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,17))));

        // v component s=1,g=1
        vreg = load_split2_spin(v+vixo(t,1,1));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,4))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,9))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,13))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,16))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,18))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_conj_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,19))));

        // v component s=1,g=2
        vreg = load_split2_spin(v+vixo(t,2,1));
        // multiply onto the field strength entry and add to result
        // the lower triangle elements are complex conjugated
        r00 = _mm256_add_pd(r00, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,5))));
        r01 = _mm256_add_pd(r01, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,10))));
        r02 = _mm256_add_pd(r02, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,14))));
        r10 = _mm256_add_pd(r10, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,17))));
        r11 = _mm256_add_pd(r11, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,19))));
        r12 = _mm256_add_pd(r12, compl_vectorreg_pointwise_mul(vreg, _mm256_loadu_pd(F+fixo(t,20))));

        // store into result tensor
        store_split2_spin(result+vixo(t,0,0), r00);
        store_split2_spin(result+vixo(t,1,0), r01);
        store_split2_spin(result+vixo(t,2,0), r02);
        store_split2_spin(result+vixo(t,0,1), r10);
        store_split2_spin(result+vixo(t,1,1), r11);
        store_split2_spin(result+vixo(t,2,1), r12);

    }

    return result_tensor;
}


}

#else
namespace qmad_history {

at::Tensor dwc_avx_tmgs_tmgsMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_tensor, double mass, double csw){
    TORCH_CHECK(0,"AVX not compiled");
    return torch::zeros({1}, v_tensor.options());
}

at::Tensor dwc_avx_mtsg_tmgsMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_tensor, double mass, double csw){
    TORCH_CHECK(0,"AVX not compiled");
    return torch::zeros({1}, v_tensor.options());
}

at::Tensor dwc_avx_mtsg_tmsgMhns (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_tensor, double mass, double csw){
    TORCH_CHECK(0,"AVX not compiled");
    return torch::zeros({1}, v_tensor.options());
}

at::Tensor dwc_avx_mtsg_tmnsgMhs_sigpre (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_tensor, double mass){
    TORCH_CHECK(0,"AVX not compiled");
    return torch::zeros({1}, v_tensor.options());
}

}
#endif

