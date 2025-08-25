// this file contains hop versions for the wilson clover
// only with mtsg memory layout
// both Fmunu and sigmaF precomputations are used

#include <torch/extension.h>
#include <omp.h>
#include <iostream>

#include "static/indexfunc_2.hpp"
#include "static/gamma_1.hpp"

namespace qmad_history {

at::Tensor dwc_hop_mtsg_tmsgMhn_fpre (const at::Tensor& U_ten, const at::Tensor& v_ten,
                                     const at::Tensor& fs_tensors,
                                     const at::Tensor& hops_ten, double mass, double csw){
                                
    // data layout for F is still t,munu,g,gi here
    
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

    TORCH_INTERNAL_ASSERT(U_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(v_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hops_ten.device().type() == at::DeviceType::CPU);

    int vol = hops_ten.size(0);

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F = fs_tensors.const_data_ptr<c10::complex<double>>();
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

                // clover term in first mu iteration
                c10::complex<double> interm (0.0,0.0);
                for (int gi = 0; gi < 3; gi++){
                    for (int munu = 0; munu < 6; munu++){
                        interm += F[fix(t,munu,g,gi)]
                                * sigf[munu][s] * v[vixo(t,gi,sigx[munu][s])];
                    }
                }
                result[vixo(t,g,s)] += -csw*0.5*interm;

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


at::Tensor dwc_hop_mtsg_tmnsgMh_sigpre (const at::Tensor& U_ten, const at::Tensor& v_ten,
                                     const at::Tensor& fs_tensors,
                                     const at::Tensor& hops_ten, double mass){
                                
    // fs_tensors is the upper half of the 2 6x6 blocks of sigma F
    
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

    TORCH_INTERNAL_ASSERT(U_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(v_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hops_ten.device().type() == at::DeviceType::CPU);

    int vol = hops_ten.size(0);

    at::Tensor result_ten = torch::empty(v_ten.sizes(), v_ten.options());
    const c10::complex<double>* U = U_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v = v_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* sF = fs_tensors.const_data_ptr<c10::complex<double>>();
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

        // clover term (grid computation)
        // iterate over the 2 triangles which correspond to spin 0,1 and 2,3 respectively
        for (int sbl = 0; sbl < 2; sbl++){
            // contribution from s=0,g=0
            c10::complex<double> v00 = v[vixo(t,0,sbl*2)];
            result[vixo(t,0,sbl*2)] += sF[sfix(t,sbl,0)]*v00;
            result[vixo(t,1,sbl*2)] += std::conj(sF[sfix(t,sbl,1)])*v00;
            result[vixo(t,2,sbl*2)] += std::conj(sF[sfix(t,sbl,2)])*v00;
            result[vixo(t,0,sbl*2+1)] += std::conj(sF[sfix(t,sbl,3)])*v00;
            result[vixo(t,1,sbl*2+1)] += std::conj(sF[sfix(t,sbl,4)])*v00;
            result[vixo(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,5)])*v00;

            // contribution from s=0,g=1
            c10::complex<double> v01 = v[vixo(t,1,sbl*2)];
            result[vixo(t,0,sbl*2)] += sF[sfix(t,sbl,1)]*v01;
            result[vixo(t,1,sbl*2)] += sF[sfix(t,sbl,6)]*v01;
            result[vixo(t,2,sbl*2)] += std::conj(sF[sfix(t,sbl,7)])*v01;
            result[vixo(t,0,sbl*2+1)] += std::conj(sF[sfix(t,sbl,8)])*v01;
            result[vixo(t,1,sbl*2+1)] += std::conj(sF[sfix(t,sbl,9)])*v01;
            result[vixo(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,10)])*v01;

            // contribution from s=0,g=2
            c10::complex<double> v02 = v[vixo(t,2,sbl*2)];
            result[vixo(t,0,sbl*2)] += sF[sfix(t,sbl,2)]*v02;
            result[vixo(t,1,sbl*2)] += sF[sfix(t,sbl,7)]*v02;
            result[vixo(t,2,sbl*2)] += sF[sfix(t,sbl,11)]*v02;
            result[vixo(t,0,sbl*2+1)] += std::conj(sF[sfix(t,sbl,12)])*v02;
            result[vixo(t,1,sbl*2+1)] += std::conj(sF[sfix(t,sbl,13)])*v02;
            result[vixo(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,14)])*v02;

            // contribution from s=1,g=0
            c10::complex<double> v10 = v[vixo(t,0,sbl*2+1)];
            result[vixo(t,0,sbl*2)] += sF[sfix(t,sbl,3)]*v10;
            result[vixo(t,1,sbl*2)] += sF[sfix(t,sbl,8)]*v10;
            result[vixo(t,2,sbl*2)] += sF[sfix(t,sbl,12)]*v10;
            result[vixo(t,0,sbl*2+1)] += sF[sfix(t,sbl,15)]*v10;
            result[vixo(t,1,sbl*2+1)] += std::conj(sF[sfix(t,sbl,16)])*v10;
            result[vixo(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,17)])*v10;

            // contribution from s=1,g=1
            c10::complex<double> v11 = v[vixo(t,1,sbl*2+1)];
            result[vixo(t,0,sbl*2)] += sF[sfix(t,sbl,4)]*v11;
            result[vixo(t,1,sbl*2)] += sF[sfix(t,sbl,9)]*v11;
            result[vixo(t,2,sbl*2)] += sF[sfix(t,sbl,13)]*v11;
            result[vixo(t,0,sbl*2+1)] += sF[sfix(t,sbl,16)]*v11;
            result[vixo(t,1,sbl*2+1)] += sF[sfix(t,sbl,18)]*v11;
            result[vixo(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,19)])*v11;

            // contribution from s=1,g=2
            c10::complex<double> v12 = v[vixo(t,2,sbl*2+1)];
            result[vixo(t,0,sbl*2)] += sF[sfix(t,sbl,5)]*v12;
            result[vixo(t,1,sbl*2)] += sF[sfix(t,sbl,10)]*v12;
            result[vixo(t,2,sbl*2)] += sF[sfix(t,sbl,14)]*v12;
            result[vixo(t,0,sbl*2+1)] += sF[sfix(t,sbl,17)]*v12;
            result[vixo(t,1,sbl*2+1)] += sF[sfix(t,sbl,19)]*v12;
            result[vixo(t,2,sbl*2+1)] += sF[sfix(t,sbl,20)]*v12;
        }
    }

    return result_ten;
}

}
