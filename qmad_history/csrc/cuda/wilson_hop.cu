#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <cuda/std/complex>

// .cuh ist die Endung für CUDA-Header, sie sind aber das gleiche wie .h
#include "indexfunc_2.cuh"
#include "gamma_1.cuh"

// I do not know how complex numbers work in Pytorch CUDA
// so I used the Pytorch C++ datatypes, but the cuda::std funciton for complex conjugate
// the test result: cuda::std does not work with this datatype

namespace qmad_history {

__global__ void dw_hop_mtsg_tmsgMh_kernel(const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, double mass, c10::complex<double> * result, int vol) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < vol){
        // mass term and mu = 0 term in same loop to minimize result access
        int mu = 0;
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vixo(t,g,s)] = (4.0 + mass) * v[vixo(t,g,s)];

                for (int gi = 0; gi < 3; gi++){
                    
                    result[vixo(t,g,s)] += (
                        std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                        * (
                            -v[vixo(hops[hix(t,mu,0)],gi,s)]
                            -gamf[mu*4+s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu*4+s])]
                        )
                        + U[uixo(t,mu,g,gi,vol)]
                        * (
                            -v[vixo(hops[hix(t,mu,1)],gi,s)]
                            +gamf[mu*4+s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu*4+s])]
                        )
                    ) * 0.5;
                }
            }
        }

        for (mu = 1; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixo(t,g,s)] += (
                            std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vixo(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu*4+s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu*4+s])]
                            )
                            + U[uixo(t,mu,g,gi,vol)]
                            * (
                                -v[vixo(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu*4+s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu*4+s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

}

at::Tensor dw_hop_mtsg_tmsgMh_cu (const at::Tensor& U_ten, const at::Tensor& v_ten,
                                  const at::Tensor& hops_ten, double mass){
    
    TORCH_CHECK(v_ten.dim() == 6);
    TORCH_CHECK(U_ten.size(1) == v_ten.size(0));
    TORCH_CHECK(U_ten.size(2) == v_ten.size(1));
    TORCH_CHECK(U_ten.size(3) == v_ten.size(2));
    TORCH_CHECK(U_ten.size(4) == v_ten.size(3));
    TORCH_CHECK(v_ten.size(4) == 4);
    TORCH_CHECK(v_ten.size(5) == 3);
    
    TORCH_CHECK(U_ten.is_contiguous());
    TORCH_CHECK(v_ten.is_contiguous());
    TORCH_CHECK(hops_ten.is_contiguous());

    TORCH_INTERNAL_ASSERT(U_ten.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(v_ten.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(hops_ten.device().type() == at::DeviceType::CUDA);

    int vol = hops_ten.size(0);

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

    // allocate one thread for each site, in 1024-thread blocks
    dw_hop_mtsg_tmsgMh_kernel<<<(vol+1023)/1024,1024>>>(U,v,hops,mass,result,vol);
    // alternatively: do not allocate more than 40 blocks (number of streaming multiprocessors)
    // int blocknum = (vol+1023)/1024;
    // if (blocknum > 40) blocknum = 40;

    return result_ten;
}


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(qmad_history, CUDA, m) {
  m.impl("dw_hop_mtsg_tmsgMh", &dw_hop_mtsg_tmsgMh_cu);
}
// muss wirklich jeder CUDA-Operator eine Variante eines C++-Operators sein?

}
