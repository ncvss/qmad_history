// only the even-odd dirac operator

#include <torch/extension.h>
#include <vector>

#ifdef PARALLELISATION_ACTIVATED
#ifndef _OPENMP
#define _OPENMP
#endif
#include <ATen/ParallelOpenMP.h>
#include <omp.h>
#endif

#include "static/indexfunc_1.hpp"
#include "static/indexfunc_2.hpp"
#include "static/gamma_1.hpp"

namespace qmad_history {

at::Tensor dw_eo_pmtsg_pxtMmghs (const at::Tensor& Ue, const at::Tensor& Uo,
                                 const at::Tensor& ve, const at::Tensor& vo,
                                 double mass, std::vector<int64_t> eodim){
    
    // I add an argument for the lattice axis dimensions of the even/odd arrays,
    // as it is easier to insert the even/odd arrays flattened
    // the even/odd have the same dimensions as normal, except that t is halved

    TORCH_CHECK(Ue.is_contiguous());
    TORCH_CHECK(Uo.is_contiguous());
    TORCH_CHECK(ve.is_contiguous());
    TORCH_CHECK(vo.is_contiguous());

    //std::cout << "the sizes of vector, gauge, and return:" << std::endl;

    // size of space-time, spin and gauge axes
    int64_t v_size [6] = {eodim[0], eodim[1], eodim[2], eodim[3], ve.size(1), ve.size(2)};

    // number of different fields and size of space-time and gauge axes
    int64_t u_size [7] = {Ue.size(0), eodim[0], eodim[1], eodim[2], eodim[3], Ue.size(2), Ue.size(3)};

    // size of the result tensor, which is even and odd stacked
    int64_t r_size [7] = {2,eodim[0], eodim[1], eodim[2], eodim[3], v_size[4], v_size[5]};

    // strides of the memory blocks
    int64_t vstride [6];
    vstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    // strides of the memory blocks
    int64_t ustride [7];
    ustride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }
    // strides of the memory blocks
    int64_t rstride [7];
    rstride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        rstride[sj] = rstride[sj+1] * r_size[sj+1];
    }

    // the output is flattened in the space-time lattice
    at::Tensor result = torch::empty({2,eodim[0]*eodim[1]*eodim[2]*eodim[3],r_size[5],r_size[6]}, ve.options());

    const c10::complex<double>* Ue_ptr = Ue.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* Uo_ptr = Uo.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* ve_ptr = ve.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* vo_ptr = vo.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();


    // teo is a variable for the additional shift in t direction
    // teo=0 if x+y+z is even on the even grid, or x+y+z is odd on the odd grid
    // teo=1 in other cases
    // the shift t+1 on the base grid is t'+teo on the eo grid
    // the shift t-1 on the base grid is t'-1+teo on the eo grid
    // to parallelise this, we have to define teo inside the x loop


    // iterate over the whole field
    // x,y,z,t are the space-time indices of U, v and result
    // s is the spin index of v and result
    // g is the gauge index of result and the first gauge index of U
    // gi is the gauge index of v and the second gauge index of U, which is summed over

    // the following is only the computation of even sites

#ifdef PARALLELISATION_ACTIVATED
    // parallelisation
    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
#else
    for (int64_t x = 0; x < v_size[0]; x++){
#endif
        int64_t teo = x%2;
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){

                    // mass term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t s = 0; s < 4; s++){
                            res_ptr[ptridx7(0,x,y,z,t,s,g,rstride)] = (4.0 + mass) * ve_ptr[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }

                    // mu = 0 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo_ptr[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -vo_ptr[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * vo_ptr[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + Ue_ptr[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo_ptr[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * vo_ptr[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo_ptr[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -vo_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * vo_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + Ue_ptr[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * vo_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo_ptr[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -vo_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * vo_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + Ue_ptr[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo_ptr[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * vo_ptr[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    // for this term, because the even and odd v are shrinked in t,
                    // we have to access different points than before:
                    // the first even and odd site in each t row have the same address on their grids
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo_ptr[ptridx7(3,x,y,z,(t-1+teo+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -vo_ptr[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * vo_ptr[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + Ue_ptr[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo_ptr[ptridx6(x,y,z,(t+teo)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * vo_ptr[ptridx6(x,y,z,(t+teo)%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                            
                }
                teo = (teo+1)%2;
            }
            teo = (teo+1)%2;
        }
    }
#ifdef PARALLELISATION_ACTIVATED
    });
#endif

    // now the odd term
    // teo again defined in the x loop

#ifdef PARALLELISATION_ACTIVATED
    // parallelisation
    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
#else
    for (int64_t x = 0; x < v_size[0]; x++){
#endif
        int64_t teo = (x+1)%2;
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){

                    // mass term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t s = 0; s < 4; s++){
                            res_ptr[ptridx7(1,x,y,z,t,s,g,rstride)] = (4.0 + mass) * vo_ptr[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }

                    // mu = 0 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue_ptr[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -ve_ptr[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * ve_ptr[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + Uo_ptr[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve_ptr[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * ve_ptr[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue_ptr[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -ve_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * ve_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + Uo_ptr[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * ve_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue_ptr[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -ve_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * ve_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + Uo_ptr[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve_ptr[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * ve_ptr[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    // for this term, because the even and odd v are shrinked in t,
                    // we have to access different points than before
                    // the first even and odd site in each t row have the same address on their grids
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue_ptr[ptridx7(3,x,y,z,(t-1+teo+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -ve_ptr[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * ve_ptr[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + Uo_ptr[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve_ptr[ptridx6(x,y,z,(t+teo)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * ve_ptr[ptridx6(x,y,z,(t+teo)%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                            
                }
                teo = (teo+1)%2;
            }
            teo = (teo+1)%2;
        }
    }
#ifdef PARALLELISATION_ACTIVATED
    });
#endif


    return result;
}


at::Tensor dw_eo_hop_pmtsg_ptMmsgh (const at::Tensor& Ue_ten, const at::Tensor& Uo_ten,
                                    const at::Tensor& ve_ten, const at::Tensor& vo_ten,
                                    const at::Tensor& hopse_ten, const at::Tensor& hopso_ten,
                                    double mass){


    // check for correct size of vector fields
    // the even/odd fields are flattened in space-time

    int eovol = hopse_ten.size(0);
    int64_t eogridsize [4] = {2, eovol, 4, 3};

    TORCH_CHECK(ve_ten.dim() == 3);
    TORCH_CHECK(eovol == ve_ten.size(0));
    TORCH_CHECK(ve_ten.size(1) == 4);
    TORCH_CHECK(ve_ten.size(2) == 3);

    TORCH_CHECK(vo_ten.dim() == 3);
    TORCH_CHECK(eovol == vo_ten.size(0));
    TORCH_CHECK(vo_ten.size(1) == 4);
    TORCH_CHECK(vo_ten.size(2) == 3);

    TORCH_CHECK(Ue_ten.dtype() == at::kComplexDouble);
    TORCH_CHECK(ve_ten.dtype() == at::kComplexDouble);
    TORCH_CHECK(Uo_ten.dtype() == at::kComplexDouble);
    TORCH_CHECK(vo_ten.dtype() == at::kComplexDouble);

    // if the data is not contiguous, we cannot calculate the pointer to its place in memory
    TORCH_CHECK(Ue_ten.is_contiguous());
    TORCH_CHECK(ve_ten.is_contiguous());
    TORCH_CHECK(Uo_ten.is_contiguous());
    TORCH_CHECK(vo_ten.is_contiguous());

    TORCH_INTERNAL_ASSERT(Ue_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(ve_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hopse_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(Uo_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(vo_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hopso_ten.device().type() == at::DeviceType::CPU);


    at::Tensor result_ten = torch::zeros(eogridsize, ve_ten.options());
    const c10::complex<double>* Ue = Ue_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* ve = ve_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops_e = hopse_ten.const_data_ptr<int32_t>();
    const c10::complex<double>* Uo = Uo_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* vo = vo_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops_o = hopso_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();


    // even lattice part
#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        
        // mass term independent
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vixo(t,g,s)] = (4.0 + mass) * ve[vixo(t,g,s)];
            }
        }
        

        for (int mu = 0; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixo(t,g,s)] += (
                            std::conj(Uo[uixo(hops_e[hix(t,mu,0)],mu,gi,g,eovol)])
                            * (
                                -vo[vixo(hops_e[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * vo[vixo(hops_e[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + Ue[uixo(t,mu,g,gi,eovol)]
                            * (
                                -vo[vixo(hops_e[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * vo[vixo(hops_e[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    // odd lattice part
#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        
        // mass term independent
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vixo(t+eovol,g,s)] = (4.0 + mass) * vo[vixo(t,g,s)];
            }
        }
        

        for (int mu = 0; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixo(t+eovol,g,s)] += (
                            std::conj(Ue[uixo(hops_o[hix(t,mu,0)],mu,gi,g,eovol)])
                            * (
                                -ve[vixo(hops_o[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * ve[vixo(hops_o[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + Uo[uixo(t,mu,g,gi,eovol)]
                            * (
                                -ve[vixo(hops_o[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * ve[vixo(hops_o[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    return result_ten;
}


at::Tensor dw_eo_hop_pmtsg_pMtmsgh (const at::Tensor& Ue_ten, const at::Tensor& Uo_ten,
                                    const at::Tensor& ve_ten, const at::Tensor& vo_ten,
                                    const at::Tensor& hopse_ten, const at::Tensor& hopso_ten,
                                    double mass){


    // check for correct size of vector fields
    // the even/odd fields are flattened in space-time

    int eovol = hopse_ten.size(0);
    int64_t eogridsize [4] = {2, eovol, 4, 3};

    TORCH_CHECK(ve_ten.dim() == 3);
    TORCH_CHECK(eovol == ve_ten.size(0));
    TORCH_CHECK(ve_ten.size(1) == 4);
    TORCH_CHECK(ve_ten.size(2) == 3);

    TORCH_CHECK(vo_ten.dim() == 3);
    TORCH_CHECK(eovol == vo_ten.size(0));
    TORCH_CHECK(vo_ten.size(1) == 4);
    TORCH_CHECK(vo_ten.size(2) == 3);

    TORCH_CHECK(Ue_ten.dtype() == at::kComplexDouble);
    TORCH_CHECK(ve_ten.dtype() == at::kComplexDouble);
    TORCH_CHECK(Uo_ten.dtype() == at::kComplexDouble);
    TORCH_CHECK(vo_ten.dtype() == at::kComplexDouble);

    // if the data is not contiguous, we cannot calculate the pointer to its place in memory
    TORCH_CHECK(Ue_ten.is_contiguous());
    TORCH_CHECK(ve_ten.is_contiguous());
    TORCH_CHECK(Uo_ten.is_contiguous());
    TORCH_CHECK(vo_ten.is_contiguous());

    TORCH_INTERNAL_ASSERT(Ue_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(ve_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hopse_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(Uo_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(vo_ten.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(hopso_ten.device().type() == at::DeviceType::CPU);


    at::Tensor result_ten = torch::zeros(eogridsize, ve_ten.options());
    const c10::complex<double>* Ue = Ue_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* ve = ve_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops_e = hopse_ten.const_data_ptr<int32_t>();
    const c10::complex<double>* Uo = Uo_ten.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* vo = vo_ten.const_data_ptr<c10::complex<double>>();
    const int32_t* hops_o = hopso_ten.const_data_ptr<int32_t>();
    c10::complex<double>* result = result_ten.mutable_data_ptr<c10::complex<double>>();


    // even lattice part
#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        
        // mass term independent
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vixo(t,g,s)] = (4.0 + mass) * ve[vixo(t,g,s)];
            }
        }
    }

#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        for (int mu = 0; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixo(t,g,s)] += (
                            std::conj(Uo[uixo(hops_e[hix(t,mu,0)],mu,gi,g,eovol)])
                            * (
                                -vo[vixo(hops_e[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * vo[vixo(hops_e[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + Ue[uixo(t,mu,g,gi,eovol)]
                            * (
                                -vo[vixo(hops_e[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * vo[vixo(hops_e[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    // odd lattice part
#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        
        // mass term independent
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vixo(t+eovol,g,s)] = (4.0 + mass) * vo[vixo(t,g,s)];
            }
        }
    }

#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        
        for (int mu = 0; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixo(t+eovol,g,s)] += (
                            std::conj(Ue[uixo(hops_o[hix(t,mu,0)],mu,gi,g,eovol)])
                            * (
                                -ve[vixo(hops_o[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * ve[vixo(hops_o[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + Uo[uixo(t,mu,g,gi,eovol)]
                            * (
                                -ve[vixo(hops_o[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * ve[vixo(hops_o[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    return result_ten;
}

}
