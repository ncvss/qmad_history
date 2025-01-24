// combine the best from both worlds: direct computation of hops
// in the template dirac operator

#include <torch/extension.h>

#ifdef VECTORISATION_ACTIVATED

#include <omp.h>
#include <immintrin.h>

#include "static/indexfunc_double.hpp"
#include "static/gamma_double.hpp"
#include "complmath_avx.hpp"
#include "load_avx.hpp"

#include "static/gamma_template.hpp"

#include <iostream>

namespace qmad_history{


// template to calculate the flattened space-time address after a hop
// (x,y,z,t) are the coordinates before the hop
// coord is the coordinate direction of the hop
// dir is 0 for a negative and 1 for a positive hop
// dsize is the length of the coord axis
// stride are the strides of the first 3 axes (the fourth one is always 1)
// this is not the actual array address, s and t are missing
// the function seems to be correct
template <int coord, int dir>
inline int hopc (int x, int y, int z, int t, int dsize, int* stride){
    if constexpr (coord == 0){
        if constexpr (dir == 0){
            return (x-1+dsize)%dsize * stride[0] + y*stride[1] + z*stride[2] + t;
        }
        if constexpr (dir == 1){
            return (x+1)%dsize * stride[0] + y*stride[1] + z*stride[2] + t;
        }
    }
    if constexpr (coord == 1){
        if constexpr (dir == 0){
            return x*stride[0] + (y-1+dsize)%dsize *stride[1] + z*stride[2] + t;
        }
        if constexpr (dir == 1){
            return x*stride[0] + (y+1)%dsize *stride[1] + z*stride[2] + t;
        }
    }
    if constexpr (coord == 2){
        if constexpr (dir == 0){
            return x*stride[0] + y*stride[1] + (z-1+dsize)%dsize *stride[2] + t;
        }
        if constexpr (dir == 1){
            return x*stride[0] + y*stride[1] + (z+1)%dsize *stride[2] + t;
        }
    }
    if constexpr (coord == 3){
        if constexpr (dir == 0){
            return x*stride[0] + y*stride[1] + z*stride[2] + (t-1+dsize)%dsize;
        }
        if constexpr (dir == 1){
            return x*stride[0] + y*stride[1] + z*stride[2] + (t+1)%dsize;
        }
    }
}

inline int adr (int x, int y, int z, int t, int* stride){
    return x*stride[0] + y*stride[1] + z*stride[2] + t;
}

// template for the body of the t,mu,g,s loop in dw_call_256d_om_template
// mu, g and s are template parameters so that the loop body can differ between iterations
// without having to check at runtime, instead generating the different code at compile time
// also, now gamma works as a template function too
// t is a function parameter, as it varies at compile time, also the loop does not change with t
template <int mu, int g, int s>
void dw_tempdir_mtsg_tmgsMhs_loop (const double * U, const double * v,
                                 __m256d massf_reg,
                                 double * result,
                                 int x, int y, int z, int t, int vol, int* sizes, int* strides){

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);

    // register with the change in result[t,s,g]
    __m256d incr;

    if constexpr (mu == 0){
        // mass term is the first term in the result
        incr = load_split_spin(v+vixo(adr(x,y,z,t,strides),g,s));
        incr = _mm256_mul_pd(incr,massf_reg);
    } else {
        // load result of previous computations to add to
        incr = load_split_spin(result+vixo(adr(x,y,z,t,strides),g,s));
    }


    for (int gi = 0; gi < 3; gi++){

        // v hop in negative mu * gammma
        __m256d v_Hmum_gam;
        if constexpr (mu == 0 || mu == 1){
            v_Hmum_gam = load_split_spin_sw(v+vixo(hopc<mu,0>(x,y,z,t,sizes[mu],strides),gi,gamx_pd[s]));
            // because mu=0,1: swap the 2 numbers in the register
        } else {
            v_Hmum_gam = load_split_spin(v+vixo(hopc<mu,0>(x,y,z,t,sizes[mu],strides),gi,gamx_pd[s]));
        }

        // multiply the gamma prefactor for s and s+1
        v_Hmum_gam = gamma_mul<mu,s>(v_Hmum_gam);
        

        // v hop in negative mu
        __m256d v_Hmum = load_split_spin(v+vixo(hopc<mu,0>(x,y,z,t,sizes[mu],strides),gi,s));

        // add those together
        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hopc<mu,0>(x,y,z,t,sizes[mu],strides),mu,gi,g,vol),v_Hmum);

        // v hop in positive mu * gamma
        __m256d v_Hmup_gam;
        if constexpr (mu == 0 || mu == 1){
            v_Hmup_gam = load_split_spin_sw(v+vixo(hopc<mu,1>(x,y,z,t,sizes[mu],strides),gi,gamx_pd[s]));
            // because mu=0,1: swap the 2 numbers in the register
        } else {
            v_Hmup_gam = load_split_spin(v+vixo(hopc<mu,1>(x,y,z,t,sizes[mu],strides),gi,gamx_pd[s]));
        }

        v_Hmup_gam = gamma_mul<mu,s>(v_Hmup_gam);
        

        // v hop in positive mu
        __m256d v_Hmup = load_split_spin(v+vixo(hopc<mu,1>(x,y,z,t,sizes[mu],strides),gi,s));

        // subtract those 2
        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

        // multiply U at this point onto v sum
        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(adr(x,y,z,t,strides),mu,g,gi,vol),v_Hmup);


        // add both U*v terms
        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

        // *(-0.5) and add to incr
        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

    }
    // store incr in result
    store_split_spin(result+vixo(adr(x,y,z,t,strides),g,s),incr);
    
}


at::Tensor dw_tempdir_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  double mass){

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


    int sizes[4];
    sizes[0] = U_tensor.size(1);
    sizes[1] = U_tensor.size(2);
    sizes[2] = U_tensor.size(3);
    sizes[3] = U_tensor.size(4);
    int str [] = {sizes[1]*sizes[2]*sizes[3], sizes[2]*sizes[3], sizes[3]};
    int vol = sizes[0]*sizes[1]*sizes[2]*sizes[3];

    // std::cout << "hopc: " << hopc<0,0>(1,2,3,4,sizes[0],str) << "\n";
    // std::cout << "hopc: " << hopc<0,1>(5,4,4,11,sizes[0],str) << "\n";
    // std::cout << "hopc: " << hopc<2,0>(1,2,3,4,sizes[0],str) << "\n";
    // std::cout << "hopc: " << hopc<2,1>(5,4,4,11,sizes[0],str) << "\n";
    // std::cout << "hopc: " << hopc<3,0>(1,2,3,4,sizes[0],str) << "\n";
    // std::cout << "hopc: " << hopc<3,1>(5,4,4,2,sizes[0],str) << "\n";
    // std::cout << "hopc: " << hopc<1,0>(1,2,3,4,sizes[0],str) << "\n";
    // std::cout << "hopc: " << hopc<1,1>(5,4,4,11,sizes[0],str) << "\n";
    // std::cout << "hopc: " << hopc<3,1>(5,4,4,12,sizes[0],str) << "\n";
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);


#pragma omp parallel for
    for (int x = 0; x < sizes[0]; x++){
        for (int y = 0; y < sizes[1]; y++){
            for (int z = 0; z < sizes[2]; z++){
                for (int t = 0; t < sizes[3]; t++){

                    // loop over mu=0,1,2,3 g=0,1,2 and s=0,2 manually with template
                    // the spin is vectorized, s=0,1 and s=2,3 are computed at the same time

                    dw_tempdir_mtsg_tmgsMhs_loop<0,0,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<0,0,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);

                    dw_tempdir_mtsg_tmgsMhs_loop<0,1,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<0,1,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);

                    dw_tempdir_mtsg_tmgsMhs_loop<0,2,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<0,2,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);


                    dw_tempdir_mtsg_tmgsMhs_loop<1,0,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<1,0,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);

                    dw_tempdir_mtsg_tmgsMhs_loop<1,1,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<1,1,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);

                    dw_tempdir_mtsg_tmgsMhs_loop<1,2,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<1,2,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);


                    dw_tempdir_mtsg_tmgsMhs_loop<2,0,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<2,0,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);

                    dw_tempdir_mtsg_tmgsMhs_loop<2,1,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<2,1,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);

                    dw_tempdir_mtsg_tmgsMhs_loop<2,2,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<2,2,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);


                    dw_tempdir_mtsg_tmgsMhs_loop<3,0,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<3,0,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);

                    dw_tempdir_mtsg_tmgsMhs_loop<3,1,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<3,1,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);

                    dw_tempdir_mtsg_tmgsMhs_loop<3,2,0>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                    dw_tempdir_mtsg_tmgsMhs_loop<3,2,2>(U,v,massf_reg,result,x,y,z,t,vol,sizes,str);
                }
            }
        }
    }

    return result_tensor;
}

}

#else
namespace qmad_history {

at::Tensor dw_templ_mtsg_tmgsMhs (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                  const at::Tensor& hops_tensor, double mass){
    
    TORCH_CHECK(0,"AVX not compiled");
    return torch::zeros({1}, v_tensor.options());
}

}
#endif
