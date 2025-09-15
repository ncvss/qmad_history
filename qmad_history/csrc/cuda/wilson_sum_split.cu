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


// in this file: versions of the dirac wilson where the sum is split via atomicadd
// there is only one kernel, the term it computes differs only by the indices
// the original file wilson_hop.cu stays for test purposes

namespace qmad_history {


// kernel computes one spin-colour component

__global__ void dw_kernel_tsg (const c10::complex<double> * U, const c10::complex<double> * v,
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


at::Tensor dw_hop_mtsg_cu_tsg (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

    // allocate one thread for each vector component, in 1024-thread blocks
    int threadnum = 1024;
    int blocknum = (vvol+threadnum-1)/threadnum;

    dw_kernel_tsg<<<blocknum,threadnum>>>(U,v,hops,result,mass,vol);
#ifdef ERROR_HANDLING_OUTPUT
    printf("dw_hop_mtsg_cu_tsg, dw_kernel_tsg error: ");
    printf(cudaGetErrorString(cudaPeekAtLastError()));
    printf("\n");
#endif

    return result_ten;
}



// one kernel computes the mass term
// one kernel computes a mu term in the sum of the neighbour terms

__global__ void dw_mass_kernel_tsg (const c10::complex<double> * v, c10::complex<double> * result, double mass, int vol){
    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp < vol*4*3){
        result[comp] = (4.0 + mass) * v[comp];
    }
}

__global__ void dw_neighbour_kernel_tmsg (const c10::complex<double> * U, const c10::complex<double> * v,
                              const int32_t * hops, double * result, int vol){

    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    int t = comp/48;

    if (t<vol){
        int mu = (comp%48)/12;
        int s = (comp%12)/3;
        int g = comp%3;

        c10::complex<double> incr (0.0,0.0);

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

        atomicAdd(result+vixo(t,g,s)*2,incr.real());
        atomicAdd(result+vixo(t,g,s)*2+1,incr.imag());

    }
}

at::Tensor dw_hop_mtsg_cu_Mtmsg (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();
    double * result_d = (double*) result;

    // allocate one thread for each vector component times mu, in 1024-thread blocks
    int threadnum = 1024;
    int mass_blocknum = (vvol+threadnum-1)/threadnum;
    int blocknum = (vvol*4+threadnum-1)/threadnum;

    // mass term
    dw_mass_kernel_tsg<<<mass_blocknum,threadnum>>>(v,result,mass,vol);
#ifdef ERROR_HANDLING_OUTPUT
    printf("dw_hop_mtsg_cu_Mtmsg, dw_mass_kernel_tsg error: ");
    printf(cudaGetErrorString(cudaPeekAtLastError()));
    printf("\n");
#endif
    // neighbour hop terms
    dw_neighbour_kernel_tmsg<<<blocknum,threadnum>>>(U,v,hops,result_d,vol);
#ifdef ERROR_HANDLING_OUTPUT
    printf("dw_hop_mtsg_cu_Mtmsg, dw_neighbour_kernel_tmsg error: ");
    printf(cudaGetErrorString(cudaPeekAtLastError()));
    printf("\n");
#endif

    return result_ten;
}


// one kernel computes the mass term
// one kernel computes a mu term in the sum of the neighbour terms

__global__ void dw_neighbour_kernel_tmsgh (const c10::complex<double> * U, const c10::complex<double> * v,
                              const int32_t * hops, double * result, int vol){

    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    int t = comp/144;

    if (t<vol){
        int mu = (comp%144)/36;
        int s = (comp%36)/9;
        int g = (comp%9)/3;
        int gi = comp%3;

        c10::complex<double> incr;

        incr = (
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

        atomicAdd(result+vixo(t,g,s)*2,incr.real());
        atomicAdd(result+vixo(t,g,s)*2+1,incr.imag());
    }
}

at::Tensor dw_hop_mtsg_cu_Mtmsgh (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();
    double * result_d = (double*) result;

    // allocate one thread for each vector component times mu and gi, in 1024-thread blocks
    int threadnum = 1024;
    int mass_blocknum = (vvol+threadnum-1)/threadnum;
    int blocknum = (vvol*4*3+threadnum-1)/threadnum;

    // mass term
    dw_mass_kernel_tsg<<<mass_blocknum,threadnum>>>(v,result,mass,vol);
#ifdef ERROR_HANDLING_OUTPUT
    printf("dw_hop_mtsg_cu_Mtmsgh, dw_mass_kernel_tsg error: ");
    printf(cudaGetErrorString(cudaPeekAtLastError()));
    printf("\n");
#endif
    // neighbour hop terms
    dw_neighbour_kernel_tmsgh<<<blocknum,threadnum>>>(U,v,hops,result_d,vol);
#ifdef ERROR_HANDLING_OUTPUT
    printf("dw_hop_mtsg_cu_Mtmsgh, dw_neighbour_kernel_tmsgh error: ");
    printf(cudaGetErrorString(cudaPeekAtLastError()));
    printf("\n");
#endif

    return result_ten;
}


// make the kernel with a 3d index

__global__ void dw_kernel_3d_tsg (const c10::complex<double> * U, const c10::complex<double> * v,
                              const int32_t * hops, c10::complex<double> * result, double mass, int vol){

    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t<vol){
        int s = threadIdx.y;
        int g = threadIdx.z;

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

        result[vixo(t,g,s)] = incr;
    }

}

at::Tensor dw_hop_mtsg_cu_3d_tsg (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

    // allocate one thread for each vector component, in blocks with 1020 thread
    // blocks are 3d, with 1 dimension each corresponding to 
    int threadnum = 3*4*85;
    dim3 thread_partition (85,4,3);
    int blocknum = (vvol+threadnum-1)/threadnum;

    dw_kernel_3d_tsg<<<blocknum,thread_partition>>>(U,v,hops,result,mass,vol);
#ifdef ERROR_HANDLING_OUTPUT
    printf("dw_hop_mtsg_cu_3d_tsg, dw_kernel_3d_tsg error: ");
    printf(cudaGetErrorString(cudaPeekAtLastError()));
    printf("\n");
#endif

    return result_ten;
}



// Registers CUDA implementations
TORCH_LIBRARY_IMPL(qmad_history, CUDA, m) {
    m.impl("dw_hop_mtsg_cu_tsg", &dw_hop_mtsg_cu_tsg);
    m.impl("dw_hop_mtsg_cu_Mtmsg", &dw_hop_mtsg_cu_Mtmsg);
    m.impl("dw_hop_mtsg_cu_Mtmsgh", &dw_hop_mtsg_cu_Mtmsgh);
    m.impl("dw_hop_mtsg_cu_3d_tsg", &dw_hop_mtsg_cu_3d_tsg);
}


}

