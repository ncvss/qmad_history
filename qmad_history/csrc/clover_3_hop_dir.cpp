// this file contains wilson clover versions that takes precomputed hop addresses
// without precomputation of the field strength

#include <torch/extension.h>
#include <omp.h>
#include <iostream>

#include "static/indexfunc_2.hpp"
#include "static/gamma_1.hpp"


namespace qmad_history {

// directly compute the entire field strength term
// use the component permutation version of sigma
inline void clover_direct (const c10::complex<double> *U, const c10::complex<double> *v, c10::complex<double> *result,
                           const int * hops, int vol, double csw, int t){

    
    int munu = 0;
    for (int mupr = 0; mupr < 4; mupr++){
        for (int nupr = 0; nupr < mupr; nupr++){
            
            // term for Q_munu
            int mu = mupr;
            int nu = nupr;
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        for (int gj = 0; gj < 3; gj++){
                            for (int gk = 0; gk < 3; gk++){
                                for (int gl = 0; gl < 3; gl++){
                                    result[vixo(t,g,s)] +=
                                        -csw * 0.0625 * (
                                            U[uixo(t,mu,g,gi,vol)]
                                            * U[uixo(hops[hix(t,mu,1)],nu,gi,gj,vol)]
                                            * std::conj(U[uixo(hops[hix(t,nu,1)],mu,gk,gj,vol)])
                                            * std::conj(U[uixo(t,nu,gl,gk,vol)])
                                            
                                            + U[uixo(t,nu,g,gi,vol)]
                                            * std::conj(U[uixo(hops[hix(hops[hix(t,mu,0)],nu,1)],mu,gj,gi,vol)])
                                            * std::conj(U[uixo(hops[hix(t,mu,0)],nu,gk,gj,vol)])
                                            * U[uixo(hops[hix(t,mu,0)],mu,gk,gl,vol)]
                                            
                                            + std::conj(U[uixo(hops[hix(t,nu,0)],nu,gi,g,vol)])
                                            * U[uixo(hops[hix(t,nu,0)],mu,gi,gj,vol)]
                                            * U[uixo(hops[hix(hops[hix(t,mu,1)],nu,0)],nu,gj,gk,vol)]
                                            * std::conj(U[uixo(t,mu,gl,gk,vol)])
                                            
                                            + std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                                            * std::conj(U[uixo(hops[hix(hops[hix(t,mu,0)],nu,0)],nu,gj,gi,vol)])
                                            * U[uixo(hops[hix(hops[hix(t,mu,0)],nu,0)],mu,gj,gk,vol)]
                                            * U[uixo(hops[hix(t,nu,0)],nu,gk,gl,vol)]
                                        ) * sigf[munu][s] * v[vixo(t,gl,sigx[munu][s])];
                                }
                            }
                        }
                    }
                }
            }

            // term for Q_numu
            mu = nupr;
            nu = mupr;
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        for (int gj = 0; gj < 3; gj++){
                            for (int gk = 0; gk < 3; gk++){
                                for (int gl = 0; gl < 3; gl++){
                                    result[vixo(t,g,s)] +=
                                        csw * 0.0625 * (
                                            U[uixo(t,mu,g,gi,vol)]
                                            * U[uixo(hops[hix(t,mu,1)],nu,gi,gj,vol)]
                                            * std::conj(U[uixo(hops[hix(t,nu,1)],mu,gk,gj,vol)])
                                            * std::conj(U[uixo(t,nu,gl,gk,vol)])
                                            
                                            + U[uixo(t,nu,g,gi,vol)]
                                            * std::conj(U[uixo(hops[hix(hops[hix(t,mu,0)],nu,1)],mu,gj,gi,vol)])
                                            * std::conj(U[uixo(hops[hix(t,mu,0)],nu,gk,gj,vol)])
                                            * U[uixo(hops[hix(t,mu,0)],mu,gk,gl,vol)]
                                            
                                            + std::conj(U[uixo(hops[hix(t,nu,0)],nu,gi,g,vol)])
                                            * U[uixo(hops[hix(t,nu,0)],mu,gi,gj,vol)]
                                            * U[uixo(hops[hix(hops[hix(t,mu,1)],nu,0)],nu,gj,gk,vol)]
                                            * std::conj(U[uixo(t,mu,gl,gk,vol)])
                                            
                                            + std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)])
                                            * std::conj(U[uixo(hops[hix(hops[hix(t,mu,0)],nu,0)],nu,gj,gi,vol)])
                                            * U[uixo(hops[hix(hops[hix(t,mu,0)],nu,0)],mu,gj,gk,vol)]
                                            * U[uixo(hops[hix(t,nu,0)],nu,gk,gl,vol)]
                                        ) * sigf[munu][s] * v[vixo(t,gl,sigx[munu][s])];
                                }
                            }
                        }
                    }
                }
            }
            
            munu++;
        }
    }

    
}


// try to rearrange the clover term using the distributive law
// we multiply the rightmost matrix onto the colour vector, then the one to the left, and so on
inline void clover_direct_rearr (const c10::complex<double> *U, const c10::complex<double> *v, c10::complex<double> *result,
                                 const int * hops, int vol, double csw, int t){

    // term for Q_munu
    int munu = 0;
    for (int mupr = 0; mupr < 4; mupr++){
        for (int nupr = 0; nupr < mupr; nupr++){

            // term for Q_munu
            int mu = mupr;
            int nu = nupr;
            for (int s = 0; s < 4; s++){
                // initialise with the vector
                c10::complex<double> aggr[3];
                for (int gl = 0; gl < 3; gl++){
                    aggr[gl] = -csw * 0.0625 * sigf[munu][s] * v[vixo(t,gl,sigx[munu][s])];
                }
                
                // multiply last matrix from last term
                c10::complex<double> aggr2[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr2[gk] += U[uixo(hops[hix(t,nu,0)],nu,gk,gl,vol)] * aggr[gl];
                    }
                }
                // next matrix
                c10::complex<double> aggr3[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr3[gj] += U[uixo(hops[hix(hops[hix(t,mu,0)],nu,0)],mu,gj,gk,vol)] * aggr2[gk];
                    }
                }
                c10::complex<double> aggr4[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr4[gi] += std::conj(U[uixo(hops[hix(hops[hix(t,mu,0)],nu,0)],nu,gj,gi,vol)]) * aggr3[gj];
                    }
                }
                // we can add the final contribution to the result
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vixo(t,g,s)] += std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)]) * aggr4[gi];
                    }
                }

                // now the same for the other 3 terms

                c10::complex<double> aggr5[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr5[gk] += std::conj(U[uixo(t,mu,gl,gk,vol)]) * aggr[gl];
                    }
                }
                c10::complex<double> aggr6[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr6[gj] += U[uixo(hops[hix(hops[hix(t,mu,1)],nu,0)],nu,gj,gk,vol)] * aggr5[gk];
                    }
                }
                c10::complex<double> aggr7[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr7[gi] += U[uixo(hops[hix(t,nu,0)],mu,gi,gj,vol)] * aggr6[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vixo(t,g,s)] += std::conj(U[uixo(hops[hix(t,nu,0)],nu,gi,g,vol)]) * aggr7[gi];
                    }
                }

                c10::complex<double> aggr8[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr8[gk] += U[uixo(hops[hix(t,mu,0)],mu,gk,gl,vol)] * aggr[gl];
                    }
                }
                c10::complex<double> aggr9[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr9[gj] += std::conj(U[uixo(hops[hix(t,mu,0)],nu,gk,gj,vol)]) * aggr8[gk];
                    }
                }
                c10::complex<double> aggr10[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr10[gi] += std::conj(U[uixo(hops[hix(hops[hix(t,mu,0)],nu,1)],mu,gj,gi,vol)]) * aggr9[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vixo(t,g,s)] += U[uixo(t,nu,g,gi,vol)] * aggr10[gi];
                    }
                }

                c10::complex<double> aggr11[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr11[gk] += std::conj(U[uixo(t,nu,gl,gk,vol)]) * aggr[gl];
                    }
                }
                c10::complex<double> aggr12[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr12[gj] += std::conj(U[uixo(hops[hix(t,nu,1)],mu,gk,gj,vol)]) * aggr11[gk];
                    }
                }
                c10::complex<double> aggr13[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr13[gi] += U[uixo(hops[hix(t,mu,1)],nu,gi,gj,vol)] * aggr12[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vixo(t,g,s)] += U[uixo(t,mu,g,gi,vol)] * aggr13[gi];
                    }
                }
            }

            // term for Q_numu
            mu = nupr;
            nu = mupr;
            for (int s = 0; s < 4; s++){
                // initialise with the vector
                c10::complex<double> aggr[3];
                for (int gl = 0; gl < 3; gl++){
                    aggr[gl] = -csw * 0.0625 * sigf[munu][s] * v[vixo(t,gl,sigx[munu][s])];
                }
                
                // multiply last matrix from last term
                c10::complex<double> aggr2[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr2[gk] += U[uixo(hops[hix(t,nu,0)],nu,gk,gl,vol)] * aggr[gl];
                    }
                }
                // next matrix
                c10::complex<double> aggr3[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr3[gj] += U[uixo(hops[hix(hops[hix(t,mu,0)],nu,0)],mu,gj,gk,vol)] * aggr2[gk];
                    }
                }
                c10::complex<double> aggr4[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr4[gi] += std::conj(U[uixo(hops[hix(hops[hix(t,mu,0)],nu,0)],nu,gj,gi,vol)]) * aggr3[gj];
                    }
                }
                // we can add the final contribution to the result
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vixo(t,g,s)] += -std::conj(U[uixo(hops[hix(t,mu,0)],mu,gi,g,vol)]) * aggr4[gi];
                    }
                }

                // now the same for the other 3 terms

                c10::complex<double> aggr5[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr5[gk] += std::conj(U[uixo(t,mu,gl,gk,vol)]) * aggr[gl];
                    }
                }
                c10::complex<double> aggr6[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr6[gj] += U[uixo(hops[hix(hops[hix(t,mu,1)],nu,0)],nu,gj,gk,vol)] * aggr5[gk];
                    }
                }
                c10::complex<double> aggr7[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr7[gi] += U[uixo(hops[hix(t,nu,0)],mu,gi,gj,vol)] * aggr6[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vixo(t,g,s)] += -std::conj(U[uixo(hops[hix(t,nu,0)],nu,gi,g,vol)]) * aggr7[gi];
                    }
                }

                c10::complex<double> aggr8[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr8[gk] += U[uixo(hops[hix(t,mu,0)],mu,gk,gl,vol)] * aggr[gl];
                    }
                }
                c10::complex<double> aggr9[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr9[gj] += std::conj(U[uixo(hops[hix(t,mu,0)],nu,gk,gj,vol)]) * aggr8[gk];
                    }
                }
                c10::complex<double> aggr10[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr10[gi] += std::conj(U[uixo(hops[hix(hops[hix(t,mu,0)],nu,1)],mu,gj,gi,vol)]) * aggr9[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vixo(t,g,s)] += -U[uixo(t,nu,g,gi,vol)] * aggr10[gi];
                    }
                }

                c10::complex<double> aggr11[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr11[gk] += std::conj(U[uixo(t,nu,gl,gk,vol)]) * aggr[gl];
                    }
                }
                c10::complex<double> aggr12[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr12[gj] += std::conj(U[uixo(hops[hix(t,nu,1)],mu,gk,gj,vol)]) * aggr11[gk];
                    }
                }
                c10::complex<double> aggr13[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr13[gi] += U[uixo(hops[hix(t,mu,1)],nu,gi,gj,vol)] * aggr12[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vixo(t,g,s)] += -U[uixo(t,mu,g,gi,vol)] * aggr13[gi];
                    }
                }
            }

            munu++;
        }
    }
    
}


at::Tensor dwc_hop_mtsg_tmsgMh_dir (const at::Tensor& U_ten, const at::Tensor& v_ten,
                                    const at::Tensor& hops_ten, double mass, double csw){
                                    
    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    // check for correct size of vector field
    TORCH_CHECK(v_ten.dim() == 6);
    TORCH_CHECK(U_ten.size(1) == v_ten.size(0));
    TORCH_CHECK(U_ten.size(2) == v_ten.size(1));
    TORCH_CHECK(U_ten.size(3) == v_ten.size(2));
    TORCH_CHECK(U_ten.size(4) == v_ten.size(3));
    TORCH_CHECK(v_ten.size(4) == 4);
    TORCH_CHECK(v_ten.size(5) == 3);

    TORCH_CHECK(U_ten.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_ten.dtype() == at::kComplexDouble);

    // if the data is not contiguous, we cannot calculate the pointer to its address
    TORCH_CHECK(U_ten.is_contiguous());
    TORCH_CHECK(v_ten.is_contiguous());

    TORCH_INTERNAL_ASSERT(U_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(v_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hops_ten.device().type() == at::DeviceType::CPU);

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

        clover_direct(U,v,result,hops,vol,csw,t);
    }


    return result_ten;
}


at::Tensor dwc_hop_mtsg_tmsgMh_dir_rearr (const at::Tensor& U_ten, const at::Tensor& v_ten,
                                          const at::Tensor& hops_ten, double mass, double csw){
                                    
    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    // check for correct size of vector field
    TORCH_CHECK(v_ten.dim() == 6);
    TORCH_CHECK(U_ten.size(1) == v_ten.size(0));
    TORCH_CHECK(U_ten.size(2) == v_ten.size(1));
    TORCH_CHECK(U_ten.size(3) == v_ten.size(2));
    TORCH_CHECK(U_ten.size(4) == v_ten.size(3));
    TORCH_CHECK(v_ten.size(4) == 4);
    TORCH_CHECK(v_ten.size(5) == 3);

    TORCH_CHECK(U_ten.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_ten.dtype() == at::kComplexDouble);

    // if the data is not contiguous, we cannot calculate the pointer to its address
    TORCH_CHECK(U_ten.is_contiguous());
    TORCH_CHECK(v_ten.is_contiguous());

    TORCH_INTERNAL_ASSERT(U_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(v_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hops_ten.device().type() == at::DeviceType::CPU);

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

        clover_direct_rearr(U,v,result,hops,vol,csw,t);
    }


    return result_ten;
}

}
