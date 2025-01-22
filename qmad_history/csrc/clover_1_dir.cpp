// this file has the dirac wilson clover operator that uses only torch and c++
// and tries to compute the field strength directly
// the computation is incomplete, I did it just as a speed test
// also, the parallelisation comes from torch

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

at::Tensor dwc_dir_mxtsg_false (const at::Tensor& U, const at::Tensor& v, double mass, double csw){

    // check for correct size of vector field
    TORCH_CHECK(v.dim() == 6);
    TORCH_CHECK(U.size(1) == v.size(0));
    TORCH_CHECK(U.size(2) == v.size(1));
    TORCH_CHECK(U.size(3) == v.size(2));
    TORCH_CHECK(U.size(4) == v.size(3));
    TORCH_CHECK(v.size(4) == 4);
    TORCH_CHECK(v.size(5) == 3);

    // if the data is not contiguous, we cannot calculate the pointer to
    // its place in memory
    TORCH_CHECK(U.is_contiguous());
    TORCH_CHECK(v.is_contiguous());

    // size of space-time, spin and gauge axes
    int64_t v_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U.size(sj);
    }

    // strides of the memory blocks
    int64_t vstride [6];
    vstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    int64_t ustride [7];
    ustride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::empty(v_size, v.options());

    const c10::complex<double>* U_ptr = U.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v.const_data_ptr<c10::complex<double>>();

    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();


    // iterate over the whole field
    // x,y,z,t are the space-time indices of U, v and result
    // s is the spin index of v and result
    // g is the gauge index of result and the first gauge index of U and F
    // gi is the gauge index of v and the second gauge index of U and F, which is summed over


    // parallelisation
    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){


                    // mass term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t s = 0; s < 4; s++){
                            res_ptr[ptridx6(x,y,z,t,s,g,vstride)] = (4.0 + mass) * v_ptr[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }

                    // mu = 0 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U_ptr[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * v_ptr[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + U_ptr[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * v_ptr[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U_ptr[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + U_ptr[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U_ptr[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * v_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + U_ptr[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * v_ptr[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U_ptr[ptridx7(3,x,y,z,(t-1+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + U_ptr[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // dirac wilson clover improvement
                    // here we directly compute the clover terms
                    // we start with only the H_-0 H_-1 H_+0 H_+1 term to test
                    // this is already much slower than with precomputation
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t gj = 0; gj < 3; gj++){
                                for (int64_t gk = 0; gk < 3; gk++){
                                    for (int64_t gl = 0; gl < 3; gl++){
                                        for (int64_t s = 0; s < 4; s++){
                                            res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                            -csw * 0.5 * (
                                                U_ptr[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                                * U_ptr[ptridx7(1,(x+1)%u_size[0],y,z,t,gi,gj,ustride)]
                                                * U_ptr[ptridx7(0,x,(y+1)%u_size[1],z,t,gj,gk,ustride)]
                                                * U_ptr[ptridx7(1,x,y,z,t,gk,gl,ustride)]
                                            ) * sigf[0][s] * v_ptr[ptridx6(x,y,z,t,sigx[0][s],gl,vstride)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // for (int64_t g = 0; g < 3; g++){
                    //     for (int64_t gi = 0; gi < 3; gi++){
                    //         for (int64_t s = 0; s < 4; s++){
                    //             res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                    //                 F20_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                    //                     * sigf[1][s] * v_ptr[ptridx6(x,y,z,t,sigx[1][s],gi,vstride)]
                    //                     *csw*0.5;
                    //         }
                    //     }
                    // }
                    // for (int64_t g = 0; g < 3; g++){
                    //     for (int64_t gi = 0; gi < 3; gi++){
                    //         for (int64_t s = 0; s < 4; s++){
                    //             res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                    //                 F21_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                    //                     * sigf[2][s] * v_ptr[ptridx6(x,y,z,t,sigx[2][s],gi,vstride)]
                    //                     *csw*0.5;
                    //         }
                    //     }
                    // }
                    // for (int64_t g = 0; g < 3; g++){
                    //     for (int64_t gi = 0; gi < 3; gi++){
                    //         for (int64_t s = 0; s < 4; s++){
                    //             res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                    //                 F30_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                    //                     * sigf[3][s] * v_ptr[ptridx6(x,y,z,t,sigx[3][s],gi,vstride)]
                    //                     *csw*0.5;
                    //         }
                    //     }
                    // }
                    // for (int64_t g = 0; g < 3; g++){
                    //     for (int64_t gi = 0; gi < 3; gi++){
                    //         for (int64_t s = 0; s < 4; s++){
                    //             res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                    //                 F31_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                    //                     * sigf[4][s] * v_ptr[ptridx6(x,y,z,t,sigx[4][s],gi,vstride)]
                    //                     *csw*0.5;
                    //         }
                    //     }
                    // }
                    // for (int64_t g = 0; g < 3; g++){
                    //     for (int64_t gi = 0; gi < 3; gi++){
                    //         for (int64_t s = 0; s < 4; s++){
                    //             res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                    //                 F32_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                    //                     * sigf[5][s] * v_ptr[ptridx6(x,y,z,t,sigx[5][s],gi,vstride)]
                    //                     *csw*0.5;
                    //         }
                    //     }
                    // }

                }
            }
        }
    }
    });

    return result;
}



}

