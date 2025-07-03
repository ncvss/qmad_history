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


// naive implementation: simply put the entire cpu function inside the kernel
// very bad performance
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



// second guess: split the computation into different kernels
// 1 thread is one output component now
// this means changes in the components do not work
// thus we need to compute t, g and s again

__global__ void mass_mtsg_kernel (const c10::complex<double> * v, double mass, c10::complex<double> * result, int vol){
    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp < vol*4*3){
        result[comp] = (4.0 + mass) * v[comp];
    }
}

__global__ void gaugeterms_mtsg_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, c10::complex<double> * result, int vol, int mu){

    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    int t = comp/12;

    if (t<vol){
        int sgcomp = comp%12;
        int s = sgcomp/3;
        int g = sgcomp%3;
        for (int gi = 0; gi < 3; gi++){
            result[comp] += (
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


at::Tensor dw_hop_mtsg_cuv2 (const at::Tensor& U_ten, const at::Tensor& v_ten,
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

    // mass term
    mass_mtsg_kernel<<<(vvol+1023)/1024,1024>>>(v,mass,result,vol);
    // gauge transport terms
    gaugeterms_mtsg_kernel<<<(vvol+1023)/1024,1024>>>(U,v,hops,result,vol,0);
    gaugeterms_mtsg_kernel<<<(vvol+1023)/1024,1024>>>(U,v,hops,result,vol,1);
    gaugeterms_mtsg_kernel<<<(vvol+1023)/1024,1024>>>(U,v,hops,result,vol,2);
    gaugeterms_mtsg_kernel<<<(vvol+1023)/1024,1024>>>(U,v,hops,result,vol,3);


    return result_ten;
}


// try 3 (inserted for debug): unroll the gi loop

__global__ void gaugeterms_mtsg_gi2_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, c10::complex<double> * result, int vol, int mu, int gi){

    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    int t = comp/12;

    if (t<vol){
        int sgcomp = comp%12;
        int s = sgcomp/3;
        int g = sgcomp%3;
        result[comp] += (
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


at::Tensor dw_hop_mtsg_cuv3 (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    //int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();
    // double * result_d = (double*) result;

    // allocate one thread for each vector component
    // the y thread index is the gsgi component (36 possible combinations)
    // the x thread index are different sites (28 sites, 28*36=1008 is the maximum prod that is <1024)
    // dim3 thread_partition = (28,36);
    // const int threadnum = 36*28;
    int threadnum = 1024;
    int thread_partition = threadnum;
    // int blocknum = (vol*36+threadnum-1)/threadnum;
    int blocknum = (vol*12+1023)/1024;

    // mass term
    mass_mtsg_kernel<<<(vol*12+1023)/1024,1024>>>(v,mass,result,vol);
    // gauge transport terms
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,0,0);
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,0,1);
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,0,2);

    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,1,0);
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,1,1);
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,1,2);

    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,2,0);
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,2,1);
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,2,2);

    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,3,0);
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,3,1);
    gaugeterms_mtsg_gi2_kernel<<<blocknum,thread_partition>>>(U,v,hops,result,vol,3,2);


    return result_ten;
}


// try 4: use atomic add to also unroll the gi loop
// mass term is the same
// problem: atomics do not exist for complex numbers
// however, the real and imaginary component of the output are independent
// so considering it as two doubles should work

__global__ void gaugeterms_gi_mtsg_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, double * result, int vol, int mu){

    //int t = blockIdx.x * 28 + threadIdx.x;
    int compstep = blockIdx.x * blockDim.x + threadIdx.x;
    int t = compstep/36;

    if (t<vol){
        //int sgcomp = threadIdx.y;
        int sgcomp = compstep%36;
        int s = sgcomp/9;
        int g = (sgcomp%9)/3;
        int gi = sgcomp%3;

        c10::complex<double> gi_step;

        gi_step = (
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

        atomicAdd(result+vixo(t,g,s)*2,gi_step.real());
        atomicAdd(result+vixo(t,g,s)*2+1,gi_step.imag());
        // result[vixo(t,g,s)] += gi_step;

    }
}


at::Tensor dw_hop_mtsg_cuv4 (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    //int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();
    double * result_d = (double*) result;

    // allocate one thread for each vector component
    int threadnum = 1024;
    // int blocknum = (vol*36+threadnum-1)/threadnum;
    int blocknum = (vol*36+1023)/1024;

    // mass term
    mass_mtsg_kernel<<<(vol*12+1023)/1024,1024>>>(v,mass,result,vol);
    // gauge transport terms
    gaugeterms_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,0);
    gaugeterms_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,1);
    gaugeterms_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,2);
    gaugeterms_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,3);

    return result_ten;
}



// here, the thread index is 3-dimensional (t,s,ggi)
__global__ void gaugeterms_gi3d_mtsg_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, double * result, int vol, int mu){

    int t = blockIdx.x * blockDim.z + threadIdx.z;

    if (t<vol){
        int s = threadIdx.y;
        int g = threadIdx.x/3;
        int gi = threadIdx.x%3;

        c10::complex<double> gi_step;

        gi_step = (
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

        atomicAdd(result+vixo(t,g,s)*2,gi_step.real());
        atomicAdd(result+vixo(t,g,s)*2+1,gi_step.imag());
    }
}


at::Tensor dw_hop_mtsg_cuv5 (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    double * result_d = (double*) result;

    // allocate one thread for each vector component
    // the x thread index is the ggi component (9 possible combinations)
    // the y index is s
    // the z thread index are different sites (28 sites, 28*36=1008 is the maximum prod that is <1024)
    int threadnum = 9*4*28;
    dim3 thread_partition (9,4,28);
    int blocknum = (vol*36+threadnum-1)/threadnum;

    // mass term
    mass_mtsg_kernel<<<(vol*12+1023)/1024,1024>>>(v,mass,result,vol);
    // gauge transport terms
    gaugeterms_gi3d_mtsg_kernel<<<blocknum,thread_partition>>>(U,v,hops,result_d,vol,0);
    gaugeterms_gi3d_mtsg_kernel<<<blocknum,thread_partition>>>(U,v,hops,result_d,vol,1);
    gaugeterms_gi3d_mtsg_kernel<<<blocknum,thread_partition>>>(U,v,hops,result_d,vol,2);
    gaugeterms_gi3d_mtsg_kernel<<<blocknum,thread_partition>>>(U,v,hops,result_d,vol,3);

    return result_ten;
}

// unroll the mu sum as well
__global__ void gaugeterms_gimu_mtsg_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, double * result, int vol){

    int compstep = blockIdx.x * blockDim.x + threadIdx.x;
    int t = compstep/(4*4*3*3);

    if (t<vol){
        int sgmucomp = compstep%(4*4*3*3);
        int mu = sgmucomp/36;
        int sgcomp = sgmucomp%36;
        int s = sgcomp/9;
        int g = (sgcomp%9)/3;
        int gi = sgcomp%3;

        c10::complex<double> gi_step;

        gi_step = (
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

        atomicAdd(result+vixo(t,g,s)*2,gi_step.real());
        atomicAdd(result+vixo(t,g,s)*2+1,gi_step.imag());
    }
}


at::Tensor dw_hop_mtsg_cuv6 (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    //int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();
    double * result_d = (double*) result;

    // allocate one thread for each vector component
    int threadnum = 1024;
    int blocknum = (vol*4*4*3*3+1023)/1024;

    // mass term
    mass_mtsg_kernel<<<(vol*12+1023)/1024,1024>>>(v,mass,result,vol);
    // gauge transport terms
    gaugeterms_gimu_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol);

    return result_ten;
}


// let 1 thread do multiple t and only make 40 thread blocks
// 1 block gets 1008 threads, which is a combination of 7 sites, 4 mu, 4 spin, 3 g, 3 gi
// in the loop, we increase t by 40*7, while mu, s, g, gi stay the same
__global__ void gaugeterms_gimu_tloop_mtsg_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, double * result, int vol){

    int compstep = blockIdx.x * blockDim.x + threadIdx.x;
    int t0 = compstep/(4*4*3*3);
    int stride = gridDim.x*7;
    int sgmucomp = compstep%(4*4*3*3);
    int mu = sgmucomp/36;
    int sgcomp = sgmucomp%36;
    int s = sgcomp/9;
    int g = (sgcomp%9)/3;
    int gi = sgcomp%3;

    for (int t = t0; t < vol; t+=stride){
        c10::complex<double> gi_step;

        gi_step = (
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

        atomicAdd(result+vixo(t,g,s)*2,gi_step.real());
        atomicAdd(result+vixo(t,g,s)*2+1,gi_step.imag());
    }
}


at::Tensor dw_hop_mtsg_cuv7 (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    //int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();
    double * result_d = (double*) result;

    // allocate only 40 blocks, which is the number of streaming multiprocessors
    int threadnum = 1008;
    int blocknum = 40;

    // mass term
    mass_mtsg_kernel<<<(vol*12+1023)/1024,1024>>>(v,mass,result,vol);
    // gauge transport terms
    gaugeterms_gimu_tloop_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol);

    return result_ten;
}


// do 40 thread blocks again, this time with all 1024 threads
// all threads are enough to compute 284 sites
// the upper 64 threads are not used so that the musggi component stays the same
__global__ void gaugeterms_gimu_tloop2_mtsg_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, double * result, int vol){

    int compstep = blockIdx.x * blockDim.x + threadIdx.x;
    if (compstep < 40896){
        int t0 = compstep/(4*4*3*3);
        int stride = 284;
        int sgmucomp = compstep%(4*4*3*3);
        int mu = sgmucomp/36;
        int sgcomp = sgmucomp%36;
        int s = sgcomp/9;
        int g = (sgcomp%9)/3;
        int gi = sgcomp%3;

        for (int t = t0; t < vol; t+=stride){
            c10::complex<double> gi_step;

            gi_step = (
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

            atomicAdd(result+vixo(t,g,s)*2,gi_step.real());
            atomicAdd(result+vixo(t,g,s)*2+1,gi_step.imag());
        }
    }
}


at::Tensor dw_hop_mtsg_cuv8 (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    //int vvol = vol*4*3;

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();
    double * result_d = (double*) result;

    // allocate only 40 blocks, which is the number of streaming multiprocessors
    int threadnum = 1024;
    int blocknum = 40;

    // mass term
    mass_mtsg_kernel<<<(vol*12+1023)/1024,1024>>>(v,mass,result,vol);
    // gauge transport terms
    gaugeterms_gimu_tloop2_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol);

    return result_ten;
}



// write multiple kernels for the substeps

__global__ void mass_mtsg_kernel2 (const c10::complex<double> * v, double mass, c10::complex<double> * result, int vol){
    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp < vol*4*3){
        result[comp] = (4.0 + mass) * v[comp];
    }
}

__global__ void minushop_gi_mtsg_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, double * result, int vol, int mu){

    int compstep = blockIdx.x * blockDim.x + threadIdx.x;
    int t = compstep/36;

    if (t<vol){
        int sgcomp = compstep%36;
        int s = sgcomp/9;
        int g = (sgcomp%9)/3;
        int gi = sgcomp%3;

        c10::complex<double> gi_step;

        gi_step = std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                * (
                    -v[vixo(hops[hix(t,mu,0)],gi,s)]
                    -gamf[mu*4+s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu*4+s])]
                ) * 0.5;

        atomicAdd(result+vixo(t,g,s)*2,gi_step.real());
        atomicAdd(result+vixo(t,g,s)*2+1,gi_step.imag());
    }
}

__global__ void plushop_gi_mtsg_kernel (const c10::complex<double> * U, const c10::complex<double> * v,
                                          const int32_t * hops, double * result, int vol, int mu){

    int compstep = blockIdx.x * blockDim.x + threadIdx.x;
    int t = compstep/36;

    if (t<vol){
        int sgcomp = compstep%36;
        int s = sgcomp/9;
        int g = (sgcomp%9)/3;
        int gi = sgcomp%3;

        c10::complex<double> gi_step;

        gi_step = U[uixo(t,mu,g,gi,vol)]
                * (
                    -v[vixo(hops[hix(t,mu,1)],gi,s)]
                    +gamf[mu*4+s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu*4+s])]
                ) * 0.5;

        atomicAdd(result+vixo(t,g,s)*2,gi_step.real());
        atomicAdd(result+vixo(t,g,s)*2+1,gi_step.imag());
    }
}

at::Tensor dw_hop_mtsg_cuv9 (const at::Tensor& U_ten, const at::Tensor& v_ten,
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
    double * result_d = (double*) result;

    // allocate one thread for each vector component
    int threadnum = 1024;
    // int blocknum = (vol*36+threadnum-1)/threadnum;
    int blocknum = (vol*36+1023)/1024;

    // mass term
    mass_mtsg_kernel2<<<(vol*12+1023)/1024,1024>>>(v,mass,result,vol);
    // gauge transport terms
    minushop_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,0);
    plushop_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,0);
    minushop_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,1);
    plushop_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,1);
    minushop_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,2);
    plushop_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,2);
    minushop_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,3);
    plushop_gi_mtsg_kernel<<<blocknum,threadnum>>>(U,v,hops,result_d,vol,3);

    return result_ten;
}



// Registers CUDA implementations
TORCH_LIBRARY_IMPL(qmad_history, CUDA, m) {
    m.impl("dw_hop_mtsg_tmsgMh", &dw_hop_mtsg_tmsgMh_cu);
    m.impl("dw_hop_mtsg_cuv2", &dw_hop_mtsg_cuv2);
    m.impl("dw_hop_mtsg_cuv3", &dw_hop_mtsg_cuv3);
    m.impl("dw_hop_mtsg_cuv4", &dw_hop_mtsg_cuv4);
    m.impl("dw_hop_mtsg_cuv5", &dw_hop_mtsg_cuv5);
    m.impl("dw_hop_mtsg_cuv6", &dw_hop_mtsg_cuv6);
    m.impl("dw_hop_mtsg_cuv7", &dw_hop_mtsg_cuv7);
    m.impl("dw_hop_mtsg_cuv8", &dw_hop_mtsg_cuv8);
    m.impl("dw_hop_mtsg_cuv9", &dw_hop_mtsg_cuv9);
}
// muss wirklich jeder CUDA-Operator eine Variante eines C++-Operators sein?

}
