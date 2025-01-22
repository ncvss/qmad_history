// only the even-odd dirac operator

#include <torch/extension.h>
#include <vector>

#ifndef _OPENMP
#define _OPENMP
#endif
#include <ATen/ParallelOpenMP.h>
#include <omp.h>

#include "indexfunc_1.hpp"
#include "gamma_1.hpp"

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

    // parallelisation
    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
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
    });

    // now the odd term
    // teo again defined in the x loop

    // parallelisation
    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
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
                    // the odd point following an even point in t direction has the same address
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
    });


    return result;
}

}
