// this file has the dirac wilson operators that use only c++
// and precompute the addresses for the hops (only 1 index for space-time)
// also, the parallelisation is base omp
// the code is asically c++ except for the access to torch tensors
// the memory layout is the old one

#include <torch/extension.h>
#include <omp.h>
#include <iostream>

#include "static/indexfunc_2.hpp"
#include "static/gamma_1.hpp"

namespace qmad_history {

at::Tensor dw_hop_mtsg_tMmgsh (const at::Tensor& U_ten, const at::Tensor& v_ten,
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

    int vol = hops_ten.size(0);

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){
                result[vixo(t,g,s)] = (4.0 + mass) * v[vixo(t,g,s)];
            }
        }

        for (int mu = 0; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixo(t,g,s)] += (
                            std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vixo(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uixo(t,mu,g,gi,vol)]
                            * (
                                -v[vixo(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    return result_ten;
}


at::Tensor dw_hop_mtsg_tMgshm (const at::Tensor& U_ten, const at::Tensor& v_ten,
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

    int vol = hops_ten.size(0);

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){
                result[vixo(t,g,s)] = (4.0 + mass) * v[vixo(t,g,s)];
            }
        }
        
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){
                for (int gi = 0; gi < 3; gi++){
                    for (int mu = 0; mu < 4; mu++){
                        
                        result[vixo(t,g,s)] += (
                            std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vixo(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uixo(t,mu,g,gi,vol)]
                            * (
                                -v[vixo(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    return result_ten;
}


at::Tensor dw_hop_mtsg_tmgsMh (const at::Tensor& U_ten, const at::Tensor& v_ten,
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

    int vol = hops_ten.size(0);

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

#pragma omp parallel for
    for (int t = 0; t < vol; t++){
        
        // mass term and mu = 0 term in same loop to minimize result access
        int mu = 0;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){
                result[vixo(t,g,s)] = (4.0 + mass) * v[vixo(t,g,s)];

                for (int gi = 0; gi < 3; gi++){
                    
                    result[vixo(t,g,s)] += (
                        std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                        * (
                            -v[vixo(hops[hix(t,mu,0)],gi,s)]
                            -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                        )
                        + U[uixo(t,mu,g,gi,vol)]
                        * (
                            -v[vixo(hops[hix(t,mu,1)],gi,s)]
                            +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                        )
                    ) * 0.5;
                }
            }
        }

        for (mu = 1; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixo(t,g,s)] += (
                            std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vixo(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uixo(t,mu,g,gi,vol)]
                            * (
                                -v[vixo(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    return result_ten;
}



at::Tensor dw_hop_mtsg_tmsgMh_cpu (const at::Tensor& U_ten, const at::Tensor& v_ten,
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

    TORCH_INTERNAL_ASSERT(U_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(v_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hops_ten.device().type() == at::DeviceType::CPU);

    int vol = hops_ten.size(0);

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();

    //std::cout << "\n\n\n" << U_ten.scalar_type() << "\n\n\n";

#pragma omp parallel for
    for (int t = 0; t < vol; t++){
        
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
                            -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                        )
                        + U[uixo(t,mu,g,gi,vol)]
                        * (
                            -v[vixo(hops[hix(t,mu,1)],gi,s)]
                            +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
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
                                -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uixo(t,mu,g,gi,vol)]
                            * (
                                -v[vixo(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    return result_ten;
}



// create blocks by halving the axes
// only works if axes are multiples of 2
at::Tensor dw_hop_block_mtsg_btmsgMh (const at::Tensor& U_ten, const at::Tensor& v_ten,
                                      const at::Tensor& hops_ten, double mass){
    
    // in this function, we use only the flattened space-time index!
    // The indices for the input arrays are U[mu,t,g,gi] and v[t,s,gi]
    
    int dims [4] = {U_ten.size(1), U_ten.size(2), U_ten.size(3), U_ten.size(4)};
    TORCH_CHECK(v_ten.dim() == 6);
    TORCH_CHECK(dims[0] == v_ten.size(0));
    TORCH_CHECK(dims[1] == v_ten.size(1));
    TORCH_CHECK(dims[2] == v_ten.size(2));
    TORCH_CHECK(dims[3] == v_ten.size(3));
    TORCH_CHECK(v_ten.size(4) == 4);
    TORCH_CHECK(v_ten.size(5) == 3);

    // space-time axes have to have even length for blocking to work
    TORCH_CHECK(dims[0]%2 + dims[1]%2 + dims[2]%2 + dims[3]%2 == 0,
                "Axes need to have even length for blocking");
    // TORCH_CHECK(dims[1]%2 == 0, "Axis needs to have even length for blocking");
    // TORCH_CHECK(dims[2]%2 == 0, "Axis needs to have even length for blocking");
    // TORCH_CHECK(dims[3]%2 == 0, "Axis needs to have even length for blocking");
    
    TORCH_CHECK(U_ten.is_contiguous());
    TORCH_CHECK(v_ten.is_contiguous());
    TORCH_CHECK(hops_ten.is_contiguous());

    TORCH_INTERNAL_ASSERT(U_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(v_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hops_ten.device().type() == at::DeviceType::CPU);

    int vol = hops_ten.size(0);

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops = hops_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();
    

#pragma omp parallel for collapse(4)
    for (int xblock = 0; xblock < dims[0]; xblock += dims[0]/2){
        for (int yblock = 0; yblock < dims[1]; yblock += dims[1]/2){
            for (int zblock = 0; zblock < dims[2]; zblock += dims[2]/2){
                for (int tblock = 0; tblock < dims[3]; tblock += dims[3]/2){
                    for (int x = xblock; x < xblock+dims[0]/2; x++){
                        for (int y = yblock; y < yblock+dims[1]/2; y++){
                            for (int z = zblock; z < zblock+dims[2]/2; z++){
                                for (int tout = tblock; tout < tblock+dims[3]/2; tout++){
                                    int t = x*dims[1]*dims[2]*dims[3] + y*dims[2]*dims[3] + z*dims[3] + tout;

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
                                                        -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                                                    )
                                                    + U[uixo(t,mu,g,gi,vol)]
                                                    * (
                                                        -v[vixo(hops[hix(t,mu,1)],gi,s)]
                                                        +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
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
                                                            -gamf[mu][s] * v[vixo(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                                                        )
                                                        + U[uixo(t,mu,g,gi,vol)]
                                                        * (
                                                            -v[vixo(hops[hix(t,mu,1)],gi,s)]
                                                            +gamf[mu][s] * v[vixo(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                                                        )
                                                    ) * 0.5;
                                                }
                                            }
                                        }
                                    }

                                }
                            }
                        }
                    }
                }
            }
        }
    }


    return result_ten;
}


}
