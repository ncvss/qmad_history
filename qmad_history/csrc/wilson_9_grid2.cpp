// This Dirac Wilson operator uses the memory layout of Grid
// the fastest index goes over sites that are furthest from each other in t direction
// it is as long as the SIMD width (2 sites for AVX)

// I originally wanted to use a halo: the sites at the border are copied
// so that the hops across the border do not have to wrap around
// however, for our case, this is more complicated than no halo

#include <torch/extension.h>

#ifdef VECTORISATION_ACTIVATED

#include <omp.h>
#include <immintrin.h>

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


// pass a template parameter if we are at a t boundary
// tbou=0 is the lower and tbou=1 the higher boundary, anything else is inside

template <int mu, int g, int s, int tbou>
inline void dw_grid_mtsgt2_tmgsMht_loop (const double * U, const double * v,
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
        __m256d v_Hmum_gam = load_hop_tbound<mu,0,tbou>(v+vixg(hops[hixd(t,mu,0)],gi,gamx[mu][s]));
        // multiply the gamma prefactor for
        v_Hmum_gam = gamma_mul_g<mu,s>(v_Hmum_gam);
        
        // v hop in negative mu
        __m256d v_Hmum = load_hop_tbound<mu,0,tbou>(v+vixg(hops[hixd(t,mu,0)],gi,s));

        // add those together
        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

        // Umu hop in negative mu, transposed
        __m256d U_Hmum = load_hop_tbound<mu,0,tbou>(U+uixg(hops[hixd(t,mu,0)],mu,gi,g,vol));

        // conjugate U and multiply onto v sum
        v_Hmum = compl_vectorreg_conj_pointwise_mul(v_Hmum, U_Hmum);


        // v hop in positive mu * gamma
        __m256d v_Hmup_gam = load_hop_tbound<mu,1,tbou>(v+vixg(hops[hixd(t,mu,1)],gi,gamx[mu][s]));
        // multiply gamma prefactor
        v_Hmup_gam = gamma_mul_g<mu,s>(v_Hmup_gam);
        
        // v hop in positive mu
        __m256d v_Hmup = load_hop_tbound<mu,1,tbou>(v+vixg(hops[hixd(t,mu,1)],gi,s));

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


at::Tensor dw_grid_mtsgt2_tmgsMht (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass){

    // memory layout is U[mu,x,y,z,t1,g,gi,t2] and v[x,y,z,t1,s,gi,t2]
    // t2 are the 2 sites that have the greatest distance, which are in one register

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

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4) * 2;
    int tlen = U_tensor.size(4);
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);

    // vectorization over 2 sites in time
    // thus the space-time loop goes only over half the volume

    // outer loop goes over the spatial grid points
    // this is the same as the Grid parallelisation for up to 8 threads
    // for more threads, Grid would also parallelise the t axis
    // we cannot do that, as it would interfere with the t vectorisation

#pragma omp parallel for
    for (int xyz = 0; xyz < vol/2; xyz+=tlen){

        // now the loop over the time axis
        // we treat first and last site separately

        // lower boundary of time axis
        int t1 = xyz;
        // loop over mu=0,1,2,3 g=0,1,2 and s=0,1,2,3 manually with template

        dw_grid_mtsgt2_tmgsMht_loop<0,0,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,0,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,0,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,0,3,0>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<0,1,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,1,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,1,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,1,3,0>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<0,2,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,2,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,2,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,2,3,0>(U,v,hops,massf_reg,result,t1,vol);


        dw_grid_mtsgt2_tmgsMht_loop<1,0,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,0,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,0,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,0,3,0>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<1,1,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,1,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,1,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,1,3,0>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<1,2,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,2,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,2,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,2,3,0>(U,v,hops,massf_reg,result,t1,vol);


        dw_grid_mtsgt2_tmgsMht_loop<2,0,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,0,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,0,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,0,3,0>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<2,1,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,1,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,1,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,1,3,0>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<2,2,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,2,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,2,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,2,3,0>(U,v,hops,massf_reg,result,t1,vol);


        dw_grid_mtsgt2_tmgsMht_loop<3,0,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,0,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,0,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,0,3,0>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<3,1,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,1,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,1,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,1,3,0>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<3,2,0,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,2,1,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,2,2,0>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,2,3,0>(U,v,hops,massf_reg,result,t1,vol);


        // inner part of time axis
        for (t1 = xyz+1; t1 < xyz+tlen-1; t1++){
            // loop over mu=0,1,2,3 g=0,1,2 and s=0,1,2,3 manually with template

            dw_grid_mtsgt2_tmgsMht_loop<0,0,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,0,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,0,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,0,3,2>(U,v,hops,massf_reg,result,t1,vol);

            dw_grid_mtsgt2_tmgsMht_loop<0,1,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,1,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,1,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,1,3,2>(U,v,hops,massf_reg,result,t1,vol);

            dw_grid_mtsgt2_tmgsMht_loop<0,2,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,2,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,2,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<0,2,3,2>(U,v,hops,massf_reg,result,t1,vol);


            dw_grid_mtsgt2_tmgsMht_loop<1,0,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,0,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,0,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,0,3,2>(U,v,hops,massf_reg,result,t1,vol);

            dw_grid_mtsgt2_tmgsMht_loop<1,1,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,1,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,1,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,1,3,2>(U,v,hops,massf_reg,result,t1,vol);

            dw_grid_mtsgt2_tmgsMht_loop<1,2,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,2,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,2,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<1,2,3,2>(U,v,hops,massf_reg,result,t1,vol);


            dw_grid_mtsgt2_tmgsMht_loop<2,0,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,0,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,0,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,0,3,2>(U,v,hops,massf_reg,result,t1,vol);

            dw_grid_mtsgt2_tmgsMht_loop<2,1,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,1,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,1,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,1,3,2>(U,v,hops,massf_reg,result,t1,vol);

            dw_grid_mtsgt2_tmgsMht_loop<2,2,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,2,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,2,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<2,2,3,2>(U,v,hops,massf_reg,result,t1,vol);


            dw_grid_mtsgt2_tmgsMht_loop<3,0,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,0,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,0,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,0,3,2>(U,v,hops,massf_reg,result,t1,vol);

            dw_grid_mtsgt2_tmgsMht_loop<3,1,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,1,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,1,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,1,3,2>(U,v,hops,massf_reg,result,t1,vol);

            dw_grid_mtsgt2_tmgsMht_loop<3,2,0,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,2,1,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,2,2,2>(U,v,hops,massf_reg,result,t1,vol);
            dw_grid_mtsgt2_tmgsMht_loop<3,2,3,2>(U,v,hops,massf_reg,result,t1,vol);
        }

        // upper boundary of time axis
        // this is signaled by template parameter tbou=1
        t1 = xyz+tlen-1;
        // loop over mu=0,1,2,3 g=0,1,2 and s=0,1,2,3 manually with template

        dw_grid_mtsgt2_tmgsMht_loop<0,0,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,0,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,0,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,0,3,1>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<0,1,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,1,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,1,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,1,3,1>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<0,2,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,2,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,2,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<0,2,3,1>(U,v,hops,massf_reg,result,t1,vol);


        dw_grid_mtsgt2_tmgsMht_loop<1,0,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,0,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,0,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,0,3,1>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<1,1,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,1,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,1,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,1,3,1>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<1,2,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,2,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,2,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<1,2,3,1>(U,v,hops,massf_reg,result,t1,vol);


        dw_grid_mtsgt2_tmgsMht_loop<2,0,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,0,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,0,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,0,3,1>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<2,1,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,1,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,1,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,1,3,1>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<2,2,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,2,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,2,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<2,2,3,1>(U,v,hops,massf_reg,result,t1,vol);


        dw_grid_mtsgt2_tmgsMht_loop<3,0,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,0,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,0,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,0,3,1>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<3,1,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,1,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,1,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,1,3,1>(U,v,hops,massf_reg,result,t1,vol);

        dw_grid_mtsgt2_tmgsMht_loop<3,2,0,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,2,1,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,2,2,1>(U,v,hops,massf_reg,result,t1,vol);
        dw_grid_mtsgt2_tmgsMht_loop<3,2,3,1>(U,v,hops,massf_reg,result,t1,vol);


    }

    return result_tensor;
}

}
#endif

