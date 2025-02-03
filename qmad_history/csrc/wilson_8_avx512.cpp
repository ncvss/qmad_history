// this is the same computation as the avx, but using templates for performance

#include <torch/extension.h>

#ifdef VECTORISATION_512_ACTIVATED

#include <omp.h>
#include <immintrin.h>

#include "static/indexfunc_double.hpp"
#include "load_avx512.hpp"
#include "complmath_avx512.hpp"


namespace qmad_history{

// template for the body of the t,mu,g,s loop in dw_call_256d_om_template
// mu and g are template parameters so that the loop body can differ between iterations
// without having to check at runtime, instead generating the different code at compile time
// also, now gamma works as a template function too
// t is a function parameter, as it varies at compile time, also the loop does not change with t
// all s degrees of freedom are loaded into one vector register
template <int mu, int g>
void dw_avx512_mtsg_tmgsMhs_loop (const double * U, const double * v,
                                 const int * hops, __m256d massf_reg,
                                 double * result, int t, int vol){

    // register with the change in result[t,s,g]
    __m512d incr;

    if constexpr (mu == 0){
        // mass term is the first term in the result
        incr = load_spin_g<4>(v+vixo(t,g,0));
        incr = _mm512_mul_pd(incr,massf_reg);
    } else {
        // load result of previous computations to add to
        incr = load_spin_g<4>(result+vixo(t,g,0));
    }


    for (int gi = 0; gi < 3; gi++){

        // v hop in negative mu * gammma
        __m512d v_Hmum_gam = gamma_mul_on_vectormem<mu>(v+vixo(hops[hixd(t,mu,0)],gi,0));

        // v hop in negative mu
        __m512d v_Hmum = load_spin_g<4>(v+vixo(hops[hixd(t,mu,0)],gi,0));

        // add those together
        v_Hmum = _mm512_add_pd(v_Hmum, v_Hmum_gam);

        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


        // v hop in positive mu * gamma
        __m512d v_Hmup_gam = gamma_mul_on_vectormem<mu>(v+vixo(hops[hixd(t,mu,1)],gi,0));

        // v hop in positive mu
        __m512d v_Hmup = load_spin_g<4>(v+vixo(hops[hixd(t,mu,1)],gi,0));

        // subtract those 2
        v_Hmup = _mm512_sub_pd(v_Hmup, v_Hmup_gam);

        // multiply U at this point onto v sum
        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


        // add both U*v terms
        v_Hmum = _mm512_add_pd(v_Hmum,v_Hmup);

        // *(-0.5) and add to incr
        incr = _mm512_fmadd_pd(v_Hmum,_mm512_set1_pd(-0.5),incr);

    }
    // store incr in result
    store_spin(result+vixo(t,g,0),incr);
    
}


at::Tensor dw_avx512_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass){

    // memory layout has to be U[mu,x,y,z,t,g,gi] and v[x,y,z,t,s,gi]

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

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // loop over mu=0,1,2,3 and g=0,1,2 manually with template
        // the spin is vectorized, s=0,1,2,3 are computed at the same time

        dw_avx512_mtsg_tmgsMhs_loop<0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_avx512_mtsg_tmgsMhs_loop<0,1>(U,v,hops,massf_reg,result,t,vol);
        dw_avx512_mtsg_tmgsMhs_loop<0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_avx512_mtsg_tmgsMhs_loop<1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_avx512_mtsg_tmgsMhs_loop<1,1>(U,v,hops,massf_reg,result,t,vol);
        dw_avx512_mtsg_tmgsMhs_loop<1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_avx512_mtsg_tmgsMhs_loop<2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_avx512_mtsg_tmgsMhs_loop<2,1>(U,v,hops,massf_reg,result,t,vol);
        dw_avx512_mtsg_tmgsMhs_loop<2,2>(U,v,hops,massf_reg,result,t,vol);

        dw_avx512_mtsg_tmgsMhs_loop<3,0>(U,v,hops,massf_reg,result,t,vol);
        dw_avx512_mtsg_tmgsMhs_loop<3,1>(U,v,hops,massf_reg,result,t,vol);
        dw_avx512_mtsg_tmgsMhs_loop<3,2>(U,v,hops,massf_reg,result,t,vol);

    }

    return result_tensor;
}

}

#else
namespace qmad_history {

at::Tensor dw_avx512_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass){
    
    TORCH_CHECK(0,"AVX-512 not compiled");
    return torch::zeros({1}, v_tensor.options());
}

}
#endif
