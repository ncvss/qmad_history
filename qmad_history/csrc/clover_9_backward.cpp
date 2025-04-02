// compute the backwards pass of the function dwc_grid_mtsg_tmngsMhs
// this has to be implemented as a product of the jacobian matrices
// the jacobian matrix of the succeeding computations is already given
// it has the same shape as a vector field, as the output is a vector field

// according to my derivations, it is almost exactly the normal wilson clover
// except that the gamma U terms have the inverse sign

#include <torch/extension.h>

#ifdef VECTORISATION_ACTIVATED

#include <omp.h>
#include <immintrin.h>

#include "static/indexfunc_double.hpp"
#include "static/gamma_double.hpp"
#include "complmath_avx.hpp"
#include "load_avx.hpp"

#include "static/gamma_template.hpp"

namespace qmad_history{

// load function for v for 2 spins with difference 2
inline __m256d load_split2_spin (const double * addr){
    // high part of the register should be s+1, so the address is increased by 6
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

// template for the body of the t,mu,g,s loop in dw_call_256d_om_template
// mu, g and s are template parameters so that the loop body can differ between iterations
// without having to check at runtime, instead generating the different code at compile time
// also, now gamma works as a template function too
// t is a function parameter, as it varies at compile time, also the loop does not change with t
template <int mu, int g, int s>
void dw_grid_mtsg_templ_backw_loop (const double * U, const double * v,
                                 const int * hops, __m256d massf_reg,
                                 double * result, int t, int vol){

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);

    // register with the change in result[t,s,g]
    __m256d incr;

    if constexpr (mu == 0){
        // mass term is the first term in the result
        incr = load_split_spin(v+vixo(t,g,s));
        incr = _mm256_mul_pd(incr,massf_reg);
    } else {
        // load result of previous computations to add to
        incr = load_split_spin(result+vixo(t,g,s));
    }


    for (int gi = 0; gi < 3; gi++){

        // v hop in negative mu * gammma
        __m256d v_Hmum_gam;
        if constexpr (mu == 0 || mu == 1){
            v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
            // because mu=0,1: swap the 2 numbers in the register
        } else {
            v_Hmum_gam = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
        }

        // multiply the gamma prefactor for s and s+1
        v_Hmum_gam = gamma_mul<mu,s>(v_Hmum_gam);
        

        // v hop in negative mu
        __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

        // subtract those 2 (only difference to normal)
        v_Hmum = _mm256_sub_pd(v_Hmum, v_Hmum_gam);

        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);

        // v hop in positive mu * gamma
        __m256d v_Hmup_gam;
        if constexpr (mu == 0 || mu == 1){
            v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
            // because mu=0,1: swap the 2 numbers in the register
        } else {
            v_Hmup_gam = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
        }

        v_Hmup_gam = gamma_mul<mu,s>(v_Hmup_gam);
        

        // v hop in positive mu (only difference to normal)
        __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

        // add those 2
        v_Hmup = _mm256_add_pd(v_Hmup, v_Hmup_gam);

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


void dwc_grid_mtsg_backw_clover (const double * v, const double * F,
                                    double * result, int t){
    
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


at::Tensor dwc_grid_mtsg_backw (const at::Tensor& U_tensor, const at::Tensor& grad_tensor,
                                  const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                                  double mass){

    // memory layout has to be U[mu,x,y,z,t,g,gi] and v[x,y,z,t,s,gi]

    TORCH_CHECK(grad_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(1) == grad_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == grad_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == grad_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == grad_tensor.size(3));
    TORCH_CHECK(grad_tensor.size(4) == 4);
    TORCH_CHECK(grad_tensor.size(5) == 3);

    // TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    // TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(grad_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

    at::Tensor result_tensor = torch::empty(grad_tensor.sizes(), grad_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)grad_tensor.const_data_ptr<c10::complex<double>>();
    const double* F = (double*)fs_tensors.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // loop over mu=0,1,2,3 g=0,1,2 and s=0,2 manually with template
        // the spin is vectorized, s=0,1 and s=2,3 are computed at the same time

        dw_grid_mtsg_templ_backw_loop<0,0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<0,0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_grid_mtsg_templ_backw_loop<0,1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<0,1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_grid_mtsg_templ_backw_loop<0,2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<0,2,2>(U,v,hops,massf_reg,result,t,vol);


        dw_grid_mtsg_templ_backw_loop<1,0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<1,0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_grid_mtsg_templ_backw_loop<1,1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<1,1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_grid_mtsg_templ_backw_loop<1,2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<1,2,2>(U,v,hops,massf_reg,result,t,vol);


        dw_grid_mtsg_templ_backw_loop<2,0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<2,0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_grid_mtsg_templ_backw_loop<2,1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<2,1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_grid_mtsg_templ_backw_loop<2,2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<2,2,2>(U,v,hops,massf_reg,result,t,vol);


        dw_grid_mtsg_templ_backw_loop<3,0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<3,0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_grid_mtsg_templ_backw_loop<3,1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<3,1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_grid_mtsg_templ_backw_loop<3,2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_grid_mtsg_templ_backw_loop<3,2,2>(U,v,hops,massf_reg,result,t,vol);

        dwc_grid_mtsg_backw_clover(v,F,result,t);
    }

    return result_tensor;
}

}

#endif
