// test the separate matrix wilson dirac operator
// for different loop orderings
// with neighbour addresses either precomputed or computed at runtime

#include <omp.h>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <complex>



// address for complex gauge field in layout U[mu,t,g,h]
inline int uix (int t, int mu, int g, int gi, int vol){
    return mu*vol*9 + t*9 + g*3 + gi;
}
// address for complex fermion field in layout v[t,s,h]
inline int vix (int t, int g, int s){
    return t*12 + s*3 + g;
}
// index for hops
inline int hix (int t, int h, int d){
    return t*8 + h*2 + d;
}

// function to calculate the pointer address from coordinates for tensors with 6 dimensions
inline int ptridx6 (int a, int b, int c, int d, int e, int f,
                        int* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5];
}

// function to calculate the pointer address from coordinates for tensors with 7 dimensions
inline int ptridx7 (int a, int b, int c, int d, int e, int f,
                        int g, int* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5]
           + g*stridearr[6];
}


static const int gamx [4][4] =
    {{3, 2, 1, 0},
     {3, 2, 1, 0},
     {2, 3, 0, 1},
     {2, 3, 0, 1} };

// gamf[mu][i] is the prefactor of spin component i of gammamu @ v
static const std::complex<double> gamf [4][4] =
    {{std::complex<double>(0, 1), std::complex<double>(0, 1), std::complex<double>(0,-1), std::complex<double>(0,-1)},
     { -1,   1,   1,  -1},
     {std::complex<double>(0, 1), std::complex<double>(0,-1), std::complex<double>(0,-1), std::complex<double>(0, 1)},
     {  1,   1,   1,   1} };

    


std::complex<double>* dw_dir_muggis (const std::complex<double>* U,
                                   const std::complex<double>* v,
                                   int grid[4], double mass){

    // strides of the memory blocks
    int v_size [6] = {grid[0],grid[1],grid[2],grid[3],4,3};
    int u_size [7] = {4,grid[0],grid[1],grid[2],grid[3],3,3};
    int vstride [6];
    vstride[5] = 1;
    for (int sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    int ustride [7];
    ustride[6] = 1;
    for (int sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }

    int vol = v_size[0]*v_size[1]*v_size[2]*v_size[3];

    std::complex<double>* result = new std::complex<double> [vol*v_size[4]*v_size[5]];


    // iterate over the whole field
    // x,y,z,t are the space-time indices of U, v and result
    // s is the spin index of v and result
    // g is the gauge index of result and the first gauge index of U
    // gi is the gauge index of v and the second gauge index of U, which is summed over

#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){

                    // mass term
                    for (int g = 0; g < 3; g++){
                        for (int s = 0; s < 4; s++){
                            result[ptridx6(x,y,z,t,s,g,vstride)] = (4.0 + mass) * v[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }

                    // mu = 0 term
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            for (int s = 0; s < 4; s++){
                                result[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * v[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + U[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * v[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            for (int s = 0; s < 4; s++){
                                result[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * v[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + U[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * v[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            for (int s = 0; s < 4; s++){
                                result[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * v[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + U[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * v[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            for (int s = 0; s < 4; s++){
                                result[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U[ptridx7(3,x,y,z,(t-1+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * v[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + U[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,y,z,(t+1)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * v[ptridx6(x,y,z,(t+1)%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                            
                }
            }
        }
    }

    return result;
}

std::complex<double>* dw_dir_sggimu (const std::complex<double>* U,
                                   const std::complex<double>* v,
                                   int grid[4], double mass){

    // strides of the memory blocks
    int v_size [6] = {grid[0],grid[1],grid[2],grid[3],4,3};
    int u_size [7] = {4,grid[0],grid[1],grid[2],grid[3],3,3};
    int vstride [6];
    vstride[5] = 1;
    for (int sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    int ustride [7];
    ustride[6] = 1;
    for (int sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }

    int vol = v_size[0]*v_size[1]*v_size[2]*v_size[3];

    std::complex<double>* res_ptr = new std::complex<double> [vol*v_size[4]*v_size[5]];

    // iterate over the whole field
    // x,y,z,t are the space-time indices of U, v and result
    // s is the spin index of v and result
    // g is the gauge index of result and the first gauge index of U
    // gi is the gauge index of v and the second gauge index of U, which is summed over

#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
                            // mass term
                            res_ptr[ptridx6(x,y,z,t,s,g,vstride)] = (4.0 + mass) * v[ptridx6(x,y,z,t,s,g,vstride)];
                            // hop terms written out for mu = 0, 1, 2, 3
                            // sum over gi corresponds to matrix product U_mu @ v
                            for (int gi = 0; gi < 3; gi++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)]
                                +=( // mu = 0 term
                                    std::conj(U[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * v[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + U[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * v[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )

                                    // mu = 1 term
                                    +std::conj(U[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * v[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + U[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * v[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )

                                    // mu = 2 term
                                    +std::conj(U[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * v[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + U[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * v[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )

                                    // mu = 3 term
                                    +std::conj(U[ptridx7(3,x,y,z,(t-1+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * v[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + U[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,y,z,(t+1)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * v[ptridx6(x,y,z,(t+1)%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                ) *0.5;
                            }
                        }
                    }
                }
            }
        }
    }

    return res_ptr;
}

std::complex<double>* dw_dir_musggi (const std::complex<double>* U,
                                   const std::complex<double>* v,
                                   int grid[4], double mass){

    // strides of the memory blocks
    int v_size [6] = {grid[0],grid[1],grid[2],grid[3],4,3};
    int u_size [7] = {4,grid[0],grid[1],grid[2],grid[3],3,3};
    int vstride [6];
    vstride[5] = 1;
    for (int sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    int ustride [7];
    ustride[6] = 1;
    for (int sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }

    int vol = v_size[0]*v_size[1]*v_size[2]*v_size[3];

    std::complex<double>* result = new std::complex<double> [vol*v_size[4]*v_size[5]];


    // iterate over the whole field
    // x,y,z,t are the space-time indices of U, v and result
    // s is the spin index of v and result
    // g is the gauge index of result and the first gauge index of U
    // gi is the gauge index of v and the second gauge index of U, which is summed over

#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){

                    // mu = 0 term
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
                            
                            // mass term in unrolled mu loop
                            result[ptridx6(x,y,z,t,s,g,vstride)] = (4.0 + mass) * v[ptridx6(x,y,z,t,s,g,vstride)];

                            for (int gi = 0; gi < 3; gi++){
                                result[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * v[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + U[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * v[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
                            for (int gi = 0; gi < 3; gi++){
                                result[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * v[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + U[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * v[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
                            for (int gi = 0; gi < 3; gi++){
                                result[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * v[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + U[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * v[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
                            for (int gi = 0; gi < 3; gi++){
                                result[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    std::conj(U[ptridx7(3,x,y,z,(t-1+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -v[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * v[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + U[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -v[ptridx6(x,y,z,(t+1)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * v[ptridx6(x,y,z,(t+1)%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                            
                }
            }
        }
    }

    return result;
}


std::complex<double>* dw_look_mugsgi (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for
    for (int t = 0; t < vol; t++){
        
        // mass term and mu = 0 term in same loop to minimize result access
        int mu = 0;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){
                result[vix(t,g,s)] = (4.0 + mass) * v[vix(t,g,s)];

                for (int gi = 0; gi < 3; gi++){
                    
                    result[vix(t,g,s)] += (
                        std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)])
                        * (
                            -v[vix(hops[hix(t,mu,0)],gi,s)]
                            -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                        )
                        + U[uix(t,mu,g,gi,vol)]
                        * (
                            -v[vix(hops[hix(t,mu,1)],gi,s)]
                            +gamf[mu][s] * v[vix(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                        )
                    ) * 0.5;
                }
            }
        }

        for (mu = 1; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vix(t,g,s)] += (
                            std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vix(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uix(t,mu,g,gi,vol)]
                            * (
                                -v[vix(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vix(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }


    return result;
}


std::complex<double>* dw_look_muggis (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){
                result[vix(t,g,s)] = (4.0 + mass) * v[vix(t,g,s)];
            }
        }

        for (int mu = 0; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int gi = 0; gi < 3; gi++){
                    for (int s = 0; s < 4; s++){
                    
                        result[vix(t,g,s)] += (
                            std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vix(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uix(t,mu,g,gi,vol)]
                            * (
                                -v[vix(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vix(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }


    return result;
}


std::complex<double>* dw_look_musggi (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for
    for (int t = 0; t < vol; t++){
        
        // mass term and mu = 0 term in same loop to minimize result access
        int mu = 0;
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vix(t,g,s)] = (4.0 + mass) * v[vix(t,g,s)];

                for (int gi = 0; gi < 3; gi++){
                    
                    result[vix(t,g,s)] += (
                        std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)])
                        * (
                            -v[vix(hops[hix(t,mu,0)],gi,s)]
                            -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                        )
                        + U[uix(t,mu,g,gi,vol)]
                        * (
                            -v[vix(hops[hix(t,mu,1)],gi,s)]
                            +gamf[mu][s] * v[vix(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                        )
                    ) * 0.5;
                }
            }
        }

        for (mu = 1; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vix(t,g,s)] += (
                            std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vix(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uix(t,mu,g,gi,vol)]
                            * (
                                -v[vix(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vix(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }


    return result;
}


std::complex<double>* dw_look_gsgimu (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for
    for (int t = 0; t < vol; t++){
        
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){

                result[vix(t,g,s)] = (4.0 + mass) * v[vix(t,g,s)];
                for (int gi = 0; gi < 3; gi++){
                    for (int mu = 0; mu < 4; mu++){
                        
                        result[vix(t,g,s)] += (
                            std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vix(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uix(t,mu,g,gi,vol)]
                            * (
                                -v[vix(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vix(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }


    return result;
}



int main (){
    int n_rep = 200;
    int n_warmup = 20;
    double mass = -0.5;
    std::cout << n_rep << " repetitions" << std::endl;
    std::cout << "mass = " << mass << std::endl;
    int Lxyz[4] = {4,4,2,4};

    // add result to noopt to prevent compiler from optimizing out function calls
    std::complex<double> noopt (0.0,0.0);

    // random number generators
    std::mt19937 generator(17);
    std::normal_distribution<double> normal(0.0, 0.7);

    bool verbose_hops = false;

    // in each loop iteration, one lattice axis is doubled
    for (int L_incr = 0; L_incr < 12; L_incr++){
        std::cout << "---" << std::endl;
        Lxyz[(L_incr+2)%4] *= 2;
        int Lx = Lxyz[0];
        int Ly = Lxyz[1];
        int Lz = Lxyz[2];
        int Lt = Lxyz[3];
        int vol = Lx * Ly * Lz * Lt;
        // strides of the grid
        int st [4] = {Ly*Lz*Lt, Lz*Lt, Lt, 1};

        int n_gauge = 3;
        int n_spin = 4;
        

        int uvol = vol * 4 * n_gauge * n_gauge;
        int vvol = vol * n_gauge * n_spin;

        // random numbers as pseudo fields
        // does not make a difference for performance
        std::complex<double>* U = new std::complex<double>[uvol];
        for (int i = 0; i < uvol; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            U[i] = std::complex<double> (ree,imm);
        }
        std::complex<double>* v = new std::complex<double>[vvol];
        for (int i = 0; i < vvol; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            v[i] = std::complex<double> (ree,imm);
        }


        int* hops = new int [vol*8];
        for (int x = 0; x < Lx; x++){
            for (int y = 0; y < Ly; y++){
                for (int z = 0; z < Lz; z++){
                    for (int t = 0; t < Lt; t++){
                        hops[(x*st[0]+y*st[1]+z*st[2]+t*st[3])*8]
                            = (x-1+Lx)%Lx*st[0]+y*st[1]+z*st[2]+t*st[3];
                        hops[(x*st[0]+y*st[1]+z*st[2]+t*st[3])*8+1]
                            = (x+1)%Lx*st[0]+y*st[1]+z*st[2]+t*st[3];
                        hops[(x*st[0]+y*st[1]+z*st[2]+t*st[3])*8+2]
                            = x*st[0]+(y-1+Ly)%Ly*st[1]+z*st[2]+t*st[3];
                        hops[(x*st[0]+y*st[1]+z*st[2]+t*st[3])*8+3]
                            = x*st[0]+(y+1)%Ly*st[1]+z*st[2]+t*st[3];
                        hops[(x*st[0]+y*st[1]+z*st[2]+t*st[3])*8+4]
                            = x*st[0]+y*st[1]+(z-1+Lz)%Lz*st[2]+t*st[3];
                        hops[(x*st[0]+y*st[1]+z*st[2]+t*st[3])*8+5]
                            = x*st[0]+y*st[1]+(z+1)%Lz*st[2]+t*st[3];
                        hops[(x*st[0]+y*st[1]+z*st[2]+t*st[3])*8+6]
                            = x*st[0]+y*st[1]+z*st[2]+(t-1+Lt)%Lt*st[3];
                        hops[(x*st[0]+y*st[1]+z*st[2]+t*st[3])*8+7]
                            = x*st[0]+y*st[1]+z*st[2]+(t+1)%Lt*st[3];
                    }
                }
            }
        }

        std::cout << vol << " sites" << std::endl;
        std::cout << "grid layout " << Lx << " " << Ly << " " << Lz << " " << Lt << std::endl;

        double * times_dir_muggis = new double [n_rep];
        double * times_dir_sggimu = new double [n_rep];
        double * times_dir_musggi = new double [n_rep];
        double * times_look_mugsgi = new double [n_rep];
        double * times_look_muggis = new double [n_rep];
        double * times_look_gsgimu = new double [n_rep];
        double * times_look_musggi = new double [n_rep];

        for (int n = 0; n < n_warmup; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[32] = randval;

            int incorrect = 0;

            std::complex<double> * result = dw_dir_muggis(U,v,Lxyz,mass);
            noopt += result[12];

            std::complex<double> * result2 = dw_dir_sggimu(U,v,Lxyz,mass);
            noopt += result2[23];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result2[i]-result[i]) > 0.001) incorrect++;
            }
            delete[] result2;

            std::complex<double> * result3 = dw_dir_musggi(U,v,Lxyz,mass);
            noopt += result3[25];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result3[i]-result[i]) > 0.001) incorrect++;
            }
            delete[] result3;

            std::complex<double>* result4 = dw_look_mugsgi(U,v,hops,mass,vol);
            noopt += result4[12];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result4[i]-result[i]) > 0.001) incorrect++;
            }
            delete[] result4;

            std::complex<double>* result5 = dw_look_muggis(U,v,hops,mass,vol);
            noopt += result5[8];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result5[i]-result[i]) > 0.001) incorrect++;
            }
            delete[] result5;

            std::complex<double>* result6 = dw_look_gsgimu(U,v,hops,mass,vol);
            noopt += result6[45];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result6[i]-result[i]) > 0.001) incorrect++;
            }
            delete[] result6;

            std::complex<double>* result7 = dw_look_musggi(U,v,hops,mass,vol);
            noopt += result7[12];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result7[i]-result[i]) > 0.001) incorrect++;
            }
            delete[] result7;

            if (n == 0){
                std::cout << "no. of incorrect sites for all ops: " << incorrect << std::endl;
            }
            delete[] result;
        }

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[31] = randval;

            auto start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result = dw_dir_muggis(U,v,Lxyz,mass);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_dir_muggis[n] = double(dur.count());
            noopt += result[12];
            delete[] result;
            
            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result2 = dw_dir_sggimu(U,v,Lxyz,mass);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_dir_sggimu[n] = double(dur.count());
            noopt += result2[23];
            delete[] result2;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result3 = dw_dir_musggi(U,v,Lxyz,mass);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_dir_musggi[n] = double(dur.count());
            noopt += result3[92];
            delete[] result3;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result4 = dw_look_mugsgi(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_look_mugsgi[n] = double(dur.count());
            noopt += result4[92];
            delete[] result4;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result5 = dw_look_muggis(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_look_muggis[n] = double(dur.count());
            noopt += result5[4];
            delete[] result5;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result6 = dw_look_gsgimu(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_look_gsgimu[n] = double(dur.count());
            noopt += result6[3];
            delete[] result6;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result7 = dw_look_musggi(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_look_musggi[n] = double(dur.count());
            noopt += result7[3];
            delete[] result7;
        }

        delete [] U;
        delete [] v;
        delete [] hops;


        double avg_dir_muggis = 0.0;
        double avg_dir_sggimu = 0.0;
        double avg_dir_musggi = 0.0;
        double avg_look_mugsgi = 0.0;
        double avg_look_muggis = 0.0;
        double avg_look_gsgimu = 0.0;
        double avg_look_musggi = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_dir_muggis,times_dir_muggis+n_rep);
        std::sort(times_dir_sggimu,times_dir_sggimu+n_rep);
        std::sort(times_dir_musggi,times_dir_musggi+n_rep);
        std::sort(times_look_mugsgi,times_look_mugsgi+n_rep);
        std::sort(times_look_muggis,times_look_muggis+n_rep);
        std::sort(times_look_gsgimu,times_look_gsgimu+n_rep);
        std::sort(times_look_musggi,times_look_musggi+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_dir_muggis += times_dir_muggis[n];
            avg_dir_sggimu += times_dir_sggimu[n];
            avg_dir_musggi += times_dir_musggi[n];
            avg_look_mugsgi += times_look_mugsgi[n];
            avg_look_muggis += times_look_muggis[n];
            avg_look_gsgimu += times_look_gsgimu[n];
            avg_look_musggi += times_look_musggi[n];
        }

        avg_dir_muggis /= double(n_res);
        avg_dir_sggimu /= double(n_res);
        avg_dir_musggi /= double(n_res);
        avg_look_mugsgi /= double(n_res);
        avg_look_muggis /= double(n_res);
        avg_look_gsgimu /= double(n_res);
        avg_look_musggi /= double(n_res);

        std::cout << "direct_muabal time: " << avg_dir_muggis/1000 << " µs\n";
        std::cout << "direct_alabmu time: " << avg_dir_sggimu/1000 << " µs\n";
        std::cout << "direct_mualab time: " << avg_dir_musggi/1000 << " µs\n";
        std::cout << "lookup_muaalb time: " << avg_look_mugsgi/1000 << " µs\n";
        std::cout << "lookup_muabal time: " << avg_look_muggis/1000 << " µs\n";
        std::cout << "lookup_aalbmu time: " << avg_look_gsgimu/1000 << " µs\n";
        std::cout << "lookup_mualab time: " << avg_look_musggi/1000 << " µs\n";

        delete [] times_dir_muggis;
        delete [] times_dir_sggimu;
        delete [] times_dir_musggi;
        delete [] times_look_mugsgi;
        delete [] times_look_muggis;
        delete [] times_look_gsgimu;
        delete [] times_look_musggi;
        
    }

    std::cout << "print noopt to prevent optimization: " << noopt << std::endl;
}

