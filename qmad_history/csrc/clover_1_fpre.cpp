// this file has the first dirac wilson clover operators that use only torch and c++
// and compute the hops directly, but take precomputed field strength
// also, the parallelisation comes from torch

#include <torch/extension.h>
#include <vector>

#ifndef _OPENMP
#define _OPENMP
#endif
#include <ATen/ParallelOpenMP.h>
#include <omp.h>

#include "static/indexfunc_1.hpp"
#include "static/gamma_1.hpp"


namespace qmad_history {

at::Tensor dwc_fpre_mntsg_xtsghmn (const at::Tensor& U, const at::Tensor& v,
                                   const std::vector<at::Tensor>& F,
                                   double mass, double csw){

    // check for correct size of vector field
    TORCH_CHECK(v.dim() == 6);
    TORCH_CHECK(U.size(1) == v.size(0));
    TORCH_CHECK(U.size(2) == v.size(1));
    TORCH_CHECK(U.size(3) == v.size(2));
    TORCH_CHECK(U.size(4) == v.size(3));
    TORCH_CHECK(v.size(4) == 4);
    TORCH_CHECK(v.size(5) == 3);

    TORCH_CHECK(U.is_contiguous());
    TORCH_CHECK(v.is_contiguous());
    for (int64_t f_i = 0; f_i < 6; f_i++){
        TORCH_CHECK(F[f_i].is_contiguous());
    }

    // size of space-time, spin and gauge axes
    int64_t v_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U.size(sj);
    }
    int64_t fsize [6];
    for (int64_t sj = 0; sj < 6; sj++){
        fsize[sj] = F[0].size(sj);
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
    int64_t fstride [6];
    fstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        fstride[sj] = fstride[sj+1] * fsize[sj+1];
    }

    at::Tensor result = torch::empty(v_size, v.options());

    const c10::complex<double>* U_ptr = U.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v.const_data_ptr<c10::complex<double>>();

    const c10::complex<double>* F10_ptr = F[0].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F20_ptr = F[1].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F21_ptr = F[2].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F30_ptr = F[3].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F31_ptr = F[4].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F32_ptr = F[5].const_data_ptr<c10::complex<double>>();

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
                    for (int64_t s = 0; s < v_size[4]; s++){
                        for (int64_t g = 0; g < v_size[5]; g++){
                            // mass term
                            res_ptr[ptridx6(x,y,z,t,s,g,vstride)] = (4.0 + mass) * v_ptr[ptridx6(x,y,z,t,s,g,vstride)];
                            // hop terms written out for mu = 0, 1, 2, 3
                            // sum over gi corresponds to matrix product U_mu @ v
                            for (int64_t gi = 0; gi < v_size[5]; gi++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)]
                                +=( // mu = 0 term
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

                                    // mu = 1 term
                                    +std::conj(U_ptr[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + U_ptr[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )

                                    // mu = 2 term
                                    +std::conj(U_ptr[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * v_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + U_ptr[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * v_ptr[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )

                                    // mu = 3 term
                                    +std::conj(U_ptr[ptridx7(3,x,y,z,(t-1+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + U_ptr[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                ) *0.5
                                
                                // dirac wilson clover improvement
                                -(
                                    F10_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[0][s] * v_ptr[ptridx6(x,y,z,t,sigx[0][s],gi,vstride)]
                                    + F20_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[1][s] * v_ptr[ptridx6(x,y,z,t,sigx[1][s],gi,vstride)]
                                    + F21_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[2][s] * v_ptr[ptridx6(x,y,z,t,sigx[2][s],gi,vstride)]
                                    + F30_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[3][s] * v_ptr[ptridx6(x,y,z,t,sigx[3][s],gi,vstride)]
                                    + F31_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[4][s] * v_ptr[ptridx6(x,y,z,t,sigx[4][s],gi,vstride)]
                                    +F32_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[5][s] * v_ptr[ptridx6(x,y,z,t,sigx[5][s],gi,vstride)]
                                )*csw*0.5;
                            }
                        }
                    }
                }
            }
        }
    }
    });

    return result;
}


at::Tensor dwc_fpre_mntsg_xtmghsn (const at::Tensor& U, const at::Tensor& v,
                                   const std::vector<at::Tensor>& F,
                                   double mass, double csw){

    // check for correct size of vector field
    TORCH_CHECK(v.dim() == 6);
    TORCH_CHECK(U.size(1) == v.size(0));
    TORCH_CHECK(U.size(2) == v.size(1));
    TORCH_CHECK(U.size(3) == v.size(2));
    TORCH_CHECK(U.size(4) == v.size(3));
    TORCH_CHECK(v.size(4) == 4);
    TORCH_CHECK(v.size(5) == 3);

    TORCH_CHECK(U.is_contiguous());
    TORCH_CHECK(v.is_contiguous());
    for (int64_t f_i = 0; f_i < 6; f_i++){
        TORCH_CHECK(F[f_i].is_contiguous());
    }

    // size of space-time, spin and gauge axes
    int64_t v_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U.size(sj);
    }
    int64_t fsize [6];
    for (int64_t sj = 0; sj < 6; sj++){
        fsize[sj] = F[0].size(sj);
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
    int64_t fstride [6];
    fstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        fstride[sj] = fstride[sj+1] * fsize[sj+1];
    }

    at::Tensor result = torch::empty(v_size, v.options());

    const c10::complex<double>* U_ptr = U.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v.const_data_ptr<c10::complex<double>>();

    const c10::complex<double>* F10_ptr = F[0].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F20_ptr = F[1].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F21_ptr = F[2].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F30_ptr = F[3].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F31_ptr = F[4].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F32_ptr = F[5].const_data_ptr<c10::complex<double>>();

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
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -= (
                                    F10_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[0][s] * v_ptr[ptridx6(x,y,z,t,sigx[0][s],gi,vstride)]
                                    + F20_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[1][s] * v_ptr[ptridx6(x,y,z,t,sigx[1][s],gi,vstride)]
                                    + F21_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[2][s] * v_ptr[ptridx6(x,y,z,t,sigx[2][s],gi,vstride)]
                                    + F30_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[3][s] * v_ptr[ptridx6(x,y,z,t,sigx[3][s],gi,vstride)]
                                    + F31_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[4][s] * v_ptr[ptridx6(x,y,z,t,sigx[4][s],gi,vstride)]
                                    +F32_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[5][s] * v_ptr[ptridx6(x,y,z,t,sigx[5][s],gi,vstride)]
                                )*csw*0.5;
                            }
                        }
                    }

                }
            }
        }
    }
    });

    return result;
}

at::Tensor dwc_fpre_mntsg_xtmnghs (const at::Tensor& U, const at::Tensor& v, const std::vector<at::Tensor>& F,
                     double mass, double csw){

    // check for correct size of vector field
    TORCH_CHECK(v.dim() == 6);
    TORCH_CHECK(U.size(1) == v.size(0));
    TORCH_CHECK(U.size(2) == v.size(1));
    TORCH_CHECK(U.size(3) == v.size(2));
    TORCH_CHECK(U.size(4) == v.size(3));
    TORCH_CHECK(v.size(4) == 4);
    TORCH_CHECK(v.size(5) == 3);

    TORCH_CHECK(U.is_contiguous());
    TORCH_CHECK(v.is_contiguous());
    for (int64_t f_i = 0; f_i < 6; f_i++){
        TORCH_CHECK(F[f_i].is_contiguous());
    }

    // size of space-time, spin and gauge axes
    int64_t v_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U.size(sj);
    }
    int64_t fsize [6];
    for (int64_t sj = 0; sj < 6; sj++){
        fsize[sj] = F[0].size(sj);
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
    int64_t fstride [6];
    fstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        fstride[sj] = fstride[sj+1] * fsize[sj+1];
    }

    at::Tensor result = torch::empty(v_size, v.options());

    const c10::complex<double>* U_ptr = U.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v.const_data_ptr<c10::complex<double>>();

    const c10::complex<double>* F10_ptr = F[0].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F20_ptr = F[1].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F21_ptr = F[2].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F30_ptr = F[3].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F31_ptr = F[4].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F32_ptr = F[5].const_data_ptr<c10::complex<double>>();

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
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F10_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[0][s] * v_ptr[ptridx6(x,y,z,t,sigx[0][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F20_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[1][s] * v_ptr[ptridx6(x,y,z,t,sigx[1][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F21_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[2][s] * v_ptr[ptridx6(x,y,z,t,sigx[2][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F30_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[3][s] * v_ptr[ptridx6(x,y,z,t,sigx[3][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F31_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[4][s] * v_ptr[ptridx6(x,y,z,t,sigx[4][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F32_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[5][s] * v_ptr[ptridx6(x,y,z,t,sigx[5][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }

                }
            }
        }
    }
    });

    return result;
}


at::Tensor dwc_fpre_mntsg_xtmdnghs (const at::Tensor& U, const at::Tensor& v,
                                    const std::vector<at::Tensor>& F,
                                    double mass, double csw){

    // check for correct size of vector field
    TORCH_CHECK(v.dim() == 6);
    TORCH_CHECK(U.size(1) == v.size(0));
    TORCH_CHECK(U.size(2) == v.size(1));
    TORCH_CHECK(U.size(3) == v.size(2));
    TORCH_CHECK(U.size(4) == v.size(3));
    TORCH_CHECK(v.size(4) == 4);
    TORCH_CHECK(v.size(5) == 3);

    TORCH_CHECK(U.is_contiguous());
    TORCH_CHECK(v.is_contiguous());
    for (int64_t f_i = 0; f_i < 6; f_i++){
        TORCH_CHECK(F[f_i].is_contiguous());
    }

    // size of space-time, spin and gauge axes
    int64_t v_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U.size(sj);
    }
    int64_t fsize [6];
    for (int64_t sj = 0; sj < 6; sj++){
        fsize[sj] = F[0].size(sj);
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
    int64_t fstride [6];
    fstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        fstride[sj] = fstride[sj+1] * fsize[sj+1];
    }

    at::Tensor result = torch::empty(v_size, v.options());

    const c10::complex<double>* U_ptr = U.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v.const_data_ptr<c10::complex<double>>();

    const c10::complex<double>* F10_ptr = F[0].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F20_ptr = F[1].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F21_ptr = F[2].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F30_ptr = F[3].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F31_ptr = F[4].const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* F32_ptr = F[5].const_data_ptr<c10::complex<double>>();

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

                    // mu = 0+ term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                    std::conj(U_ptr[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * v_ptr[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    ) * 0.5;
                            }
                        }
                    }

                    // mu = 0- term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                    U_ptr[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * v_ptr[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )* 0.5;
                            }
                        }
                    }

                    // mu = 1+ term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                    std::conj(U_ptr[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    ) * 0.5;
                            }
                        }
                    }

                    // mu = 1- term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                    U_ptr[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    ) * 0.5;
                            }
                        }
                    }

                    // mu = 2+ term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                    std::conj(U_ptr[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * v_ptr[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    ) * 0.5;
                            }
                        }
                    }

                    // mu = 2- term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                    U_ptr[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * v_ptr[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    ) * 0.5;
                            }
                        }
                    }

                    // mu = 3+ term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                    std::conj(U_ptr[ptridx7(3,x,y,z,(t-1+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    ) * 0.5;
                            }
                        }
                    }

                    // mu = 3- term
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] +=
                                    U_ptr[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],gamx[3][s],gi,vstride)]
                                    ) * 0.5;
                            }
                        }
                    }


                    // dirac wilson clover improvement
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F10_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[0][s] * v_ptr[ptridx6(x,y,z,t,sigx[0][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F20_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[1][s] * v_ptr[ptridx6(x,y,z,t,sigx[1][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F21_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[2][s] * v_ptr[ptridx6(x,y,z,t,sigx[2][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F30_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[3][s] * v_ptr[ptridx6(x,y,z,t,sigx[3][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F31_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[4][s] * v_ptr[ptridx6(x,y,z,t,sigx[4][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t gi = 0; gi < 3; gi++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] -=
                                    F32_ptr[ptridx6(x,y,z,t,g,gi,fstride)]
                                        * sigf[5][s] * v_ptr[ptridx6(x,y,z,t,sigx[5][s],gi,vstride)]
                                        *csw*0.5;
                            }
                        }
                    }

                }
            }
        }
    }
    });

    return result;
}

}
