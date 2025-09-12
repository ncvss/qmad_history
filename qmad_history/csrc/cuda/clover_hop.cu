#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <cuda/std/complex>

// .cuh ist die Endung für CUDA-Header, sie sind aber das gleiche wie .h
#include "indexfunc_2.cuh"
#include "gamma_1.cuh"

// in this file: versions of the dirac wilson clover

namespace qmad_history {

// kernel computes one spin-colour component
// using the F_munu precomputation

__global__ void dwc_kernel_tsg_fpre (const c10::complex<double> * U, const c10::complex<double> * v, const c10::complex<double> * F,
                              const int32_t * hops, c10::complex<double> * result, double mass, double csw, int vol){

    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    int t = comp/12;

    if (t<vol){
        int s = (comp%12)/3;
        int g = comp%3;

        c10::complex<double> incr = (4.0 + mass) * v[vixo(t,g,s)];
        for (int mu = 0; mu < 4; mu++){
            for (int gi = 0; gi < 3; gi++){
                incr += (
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

        c10::complex<double> cl_incr (0.0,0.0);
        for (int gi = 0; gi < 3; gi++){
            for (int munu = 0; munu < 6; munu++){
                cl_incr += F[fix(t,munu,g,gi)]
                        * sigf[munu*4+s] * v[vixo(t,gi,sigx[munu*4+s])];
            }
        }

        result[vixo(t,g,s)] =  - csw*0.5*cl_incr;

    }

}


at::Tensor dwc_hop_mtsg_cu_tsg_fpre (const at::Tensor& U_ten, const at::Tensor& v_ten, const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_ten, double mass, double csw){

    printf("call: dwc_hop_mtsg_cu_tsg_fpre\n");
    
    TORCH_CHECK(v_ten.dim() == 6);
    TORCH_CHECK(U_ten.size(1) == v_ten.size(0));
    TORCH_CHECK(U_ten.size(2) == v_ten.size(1));
    TORCH_CHECK(U_ten.size(3) == v_ten.size(2));
    TORCH_CHECK(U_ten.size(4) == v_ten.size(3));
    TORCH_CHECK(v_ten.size(4) == 4);
    TORCH_CHECK(v_ten.size(5) == 3);
    
    TORCH_CHECK(U_ten.is_contiguous());
    TORCH_CHECK(v_ten.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());
    TORCH_CHECK(hops_ten.is_contiguous());

    TORCH_INTERNAL_ASSERT(U_ten.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(v_ten.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(hops_ten.device().type() == at::DeviceType::CUDA);

    int vol = hops_ten.size(0);
    int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F = fs_tensors.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

    // allocate one thread for each vector component, in 1024-thread blocks
    int threadnum = 1024;
    int blocknum = (vvol+threadnum-1)/threadnum;

    // printf("threadnum vs lattice: %d %d\n", vol, threadnum*blocknum);

    printf("before kernel: dwc_hop_mtsg_cu_tsg_fpre\n");

    dwc_kernel_tsg_fpre<<<blocknum,threadnum>>>(U,v,F,hops,result,mass,csw,vol);
    //checkCudaErrors(cudaGetLastError());


    return result_ten;
}




// kernel computes one spin-colour component for wilson contribution
// the sigma F precompute clover cannot be split into components, so it needs its own kernel
// clover can however be split into the two blocks

__global__ void dwc_w_kernel_tsg (const c10::complex<double> * U, const c10::complex<double> * v,
                              const int32_t * hops, c10::complex<double> * result, double mass, int vol){

    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    int t = comp/12;

    if (t<vol){
        int s = (comp%12)/3;
        int g = comp%3;

        c10::complex<double> incr = (4.0 + mass) * v[comp];

        for (int mu = 0; mu < 4; mu++){
            for (int gi = 0; gi < 3; gi++){
                incr += (
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

        result[comp] = incr;

    }

}


__global__ void dwc_cl_kernel_sigpre (const c10::complex<double> * v, const c10::complex<double> * sF,
                                      c10::complex<double> * result, int vol){
    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    int t = comp/2;

    if (t<vol){
        int sbl = comp%2;

        c10::complex<double> r00 (0.0,0.0);
        c10::complex<double> r10 (0.0,0.0);
        c10::complex<double> r20 (0.0,0.0);
        c10::complex<double> r01 (0.0,0.0);
        c10::complex<double> r11 (0.0,0.0);
        c10::complex<double> r21 (0.0,0.0);

        // contribution from s=0,g=0
        c10::complex<double> v00 = v[vixo(t,0,sbl*2)];
        r00 += sF[sfix(t,sbl,0)]*v00;
        r10 += std::conj(sF[sfix(t,sbl,1)])*v00;
        r20 += std::conj(sF[sfix(t,sbl,2)])*v00;
        r01 += std::conj(sF[sfix(t,sbl,3)])*v00;
        r11 += std::conj(sF[sfix(t,sbl,4)])*v00;
        r21 += std::conj(sF[sfix(t,sbl,5)])*v00;

        // contribution from s=0,g=1
        c10::complex<double> v01 = v[vixo(t,1,sbl*2)];
        r00 += sF[sfix(t,sbl,1)]*v01;
        r10 += sF[sfix(t,sbl,6)]*v01;
        r20 += std::conj(sF[sfix(t,sbl,7)])*v01;
        r01 += std::conj(sF[sfix(t,sbl,8)])*v01;
        r11 += std::conj(sF[sfix(t,sbl,9)])*v01;
        r21 += std::conj(sF[sfix(t,sbl,10)])*v01;

        // contribution from s=0,g=2
        c10::complex<double> v02 = v[vixo(t,2,sbl*2)];
        r00 += sF[sfix(t,sbl,2)]*v02;
        r10 += sF[sfix(t,sbl,7)]*v02;
        r20 += sF[sfix(t,sbl,11)]*v02;
        r01 += std::conj(sF[sfix(t,sbl,12)])*v02;
        r11 += std::conj(sF[sfix(t,sbl,13)])*v02;
        r21 += std::conj(sF[sfix(t,sbl,14)])*v02;

        // contribution from s=1,g=0
        c10::complex<double> v10 = v[vixo(t,0,sbl*2+1)];
        r00 += sF[sfix(t,sbl,3)]*v10;
        r10 += sF[sfix(t,sbl,8)]*v10;
        r20 += sF[sfix(t,sbl,12)]*v10;
        r01 += sF[sfix(t,sbl,15)]*v10;
        r11 += std::conj(sF[sfix(t,sbl,16)])*v10;
        r21 += std::conj(sF[sfix(t,sbl,17)])*v10;

        // contribution from s=1,g=1
        c10::complex<double> v11 = v[vixo(t,1,sbl*2+1)];
        r00 += sF[sfix(t,sbl,4)]*v11;
        r10 += sF[sfix(t,sbl,9)]*v11;
        r20 += sF[sfix(t,sbl,13)]*v11;
        r01 += sF[sfix(t,sbl,16)]*v11;
        r11 += sF[sfix(t,sbl,18)]*v11;
        r21 += std::conj(sF[sfix(t,sbl,19)])*v11;

        // contribution from s=1,g=2
        c10::complex<double> v12 = v[vixo(t,2,sbl*2+1)];
        r00 += sF[sfix(t,sbl,5)]*v12;
        r10 += sF[sfix(t,sbl,10)]*v12;
        r20 += sF[sfix(t,sbl,14)]*v12;
        r01 += sF[sfix(t,sbl,17)]*v12;
        r11 += sF[sfix(t,sbl,19)]*v12;
        r21 += sF[sfix(t,sbl,20)]*v12;

        result[vixo(t,0,sbl*2)] += r00;
        result[vixo(t,1,sbl*2)] += r10;
        result[vixo(t,2,sbl*2)] += r20;
        result[vixo(t,0,sbl*2+1)] += r01;
        result[vixo(t,1,sbl*2+1)] += r11;
        result[vixo(t,2,sbl*2+1)] += r21;

    }

}


at::Tensor dwc_hop_mtsg_cu_tsg_sigpre (const at::Tensor& U_ten, const at::Tensor& v_ten, const at::Tensor& fs_tensors,
                                  const at::Tensor& hops_ten, double mass){
    
    printf("call: dwc_hop_mtsg_cu_tsg_sigpre\n");

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
    int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F = fs_tensors.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

    // allocate one thread for each vector component, in 1024-thread blocks
    int threadnum = 1024;
    int blocknum = (vvol+threadnum-1)/threadnum;
    // for clover, only 2 threads per site, one for each 6x6 block
    int cl_blocknum = (vol*2+threadnum-1)/threadnum;

    printf("before kernel: dwc_hop_mtsg_cu_tsg_sigpre\n");

    dwc_w_kernel_tsg<<<blocknum,threadnum>>>(U,v,hops,result,mass,vol);
    //checkCudaErrors(cudaGetLastError());
    dwc_cl_kernel_sigpre<<<cl_blocknum,threadnum>>>(v,F,result,vol);
    //checkCudaErrors(cudaGetLastError());

    return result_ten;
}


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(qmad_history, CUDA, m) {
    m.impl("dwc_hop_mtsg_cu_tsg_fpre", &dwc_hop_mtsg_cu_tsg_fpre);
    m.impl("dwc_hop_mtsg_cu_tsg_sigpre", &dwc_hop_mtsg_cu_tsg_sigpre);
}

}
