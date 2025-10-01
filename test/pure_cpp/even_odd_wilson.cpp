// test performance of wilson dirac operator with even-odd split lattice
// we also compare lookup tables for the addresses to runtime computations
// the lattice gets halved along the t axis
// then the sites are classified like:

// xyz\t:  0  1  2  3  4  5
// 0    : e0 o0 e1 o1 e2 o2
// 1    : o0 e0 o1 e1 o2 e2
// 3    : e0 o0 e1 o1 e2 o2
// 4    : o0 e0 o1 e1 o2 e2

// a step in t direction on the even/odd lattice is t+teo
// teo=(x+y+z+eo)%2, eo is the parity of the lattice

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




// separate matrices wilson dirac operator with runtime computed hops on full grid
std::complex<double>* dw_direct_musggi (const std::complex<double>* U,
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
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
                            result[ptridx6(x,y,z,t,s,g,vstride)] = (4.0 + mass) * v[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }

                    // mu = 0 term
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
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



// separate matrices wilson dirac operator with precomputed hops on full grid
std::complex<double>* dw_look_musggi (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index!
    // The indices for the input arrays are U[mu,t,g,gi] and v[t,s,gi]

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


// separate matrices wilson dirac operator with runtime computed hops on even-odd grid
// the mass term and neighbour terms have the same t loop
std::complex<double>* dw_eo_dir_pmusggi (const std::complex<double>* Ue, const std::complex<double>* Uo,
                                 const std::complex<double>* ve, std::complex<double>* vo,
                                 double mass, int eodim[4]){
    
    // the even/odd lattices have the same dimensions as the full one, except that the t axis is halved

    // size of space-time, spin and gauge axes
    int v_size [6] = {eodim[0], eodim[1], eodim[2], eodim[3], 4, 3};

    // number of different fields and size of space-time and gauge axes
    int u_size [7] = {4, eodim[0], eodim[1], eodim[2], eodim[3], 3, 3};

    // size of the result tensor, which is even and odd stacked
    int r_size [7] = {2,eodim[0], eodim[1], eodim[2], eodim[3], 4, 3};

    // strides of the memory blocks
    int vstride [6];
    vstride[5] = 1;
    for (int sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    // strides of the memory blocks
    int ustride [7];
    ustride[6] = 1;
    for (int sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }
    // strides of the memory blocks
    int rstride [7];
    rstride[6] = 1;
    for (int sj = 5; sj >= 0; sj--){
        rstride[sj] = rstride[sj+1] * r_size[sj+1];
    }

    // the output is flattened in the space-time lattice
    std::complex<double>* result = new std::complex<double> [2*eodim[0]*eodim[1]*eodim[2]*eodim[3]*4*3];


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

#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        int teo = x%2;
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){

                    // mass term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        
                            result[ptridx7(0,x,y,z,t,s,g,rstride)] = (4.0 + mass) * ve[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }

                    // mu = 0 term
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
                            for (int gi = 0; gi < 3; gi++){
                                result[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -vo[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * vo[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + Ue[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * vo[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -vo[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * vo[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + Ue[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * vo[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -vo[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * vo[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + Ue[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * vo[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    // for this term, because the even and odd v are shrinked in t,
                    // we have to access different points than before:
                    // the first even and odd site in each t row have the same address on their grids
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo[ptridx7(3,x,y,z,(t-1+teo+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -vo[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * vo[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + Ue[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo[ptridx6(x,y,z,(t+teo)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * vo[ptridx6(x,y,z,(t+teo)%v_size[3],gamx[3][s],gi,vstride)]
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


    // now the odd term
    // teo again defined in the x loop

#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        int teo = (x+1)%2;
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){

                    // mass term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        
                            result[ptridx7(1,x,y,z,t,s,g,rstride)] = (4.0 + mass) * vo[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }

                    // mu = 0 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -ve[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * ve[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + Uo[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * ve[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -ve[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * ve[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + Uo[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * ve[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -ve[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * ve[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + Uo[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * ve[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    // for this term, because the even and odd v are shrinked in t,
                    // we have to access different points than before
                    // the first even and odd site in each t row have the same address on their grids
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue[ptridx7(3,x,y,z,(t-1+teo+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -ve[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * ve[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + Uo[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve[ptridx6(x,y,z,(t+teo)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * ve[ptridx6(x,y,z,(t+teo)%v_size[3],gamx[3][s],gi,vstride)]
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



    return result;
}


// separate matrices wilson dirac operator with precomputed hops on even-odd grid
// the mass term and neighbour terms have the same t loop
std::complex<double>* dw_eo_look_pmusggi (const std::complex<double>* Ue, const std::complex<double>* Uo,
                                          const std::complex<double>* ve, const std::complex<double>* vo,
                                          const int* hops_e, const int* hops_o, double mass, int eovol){


    std::complex<double>* result = new std::complex<double> [2*eovol*4*3];

// even lattice part
#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        
        // mass term independent
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vix(t,g,s)] = (4.0 + mass) * ve[vix(t,g,s)];
            }
        }
        

        for (int mu = 0; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vix(t,g,s)] += (
                            std::conj(Uo[uix(hops_e[hix(t,mu,0)],mu,gi,g,eovol)])
                            * (
                                -vo[vix(hops_e[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * vo[vix(hops_e[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + Ue[uix(t,mu,g,gi,eovol)]
                            * (
                                -vo[vix(hops_e[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * vo[vix(hops_e[hix(t,mu,1)],gi,gamx[mu][s])]
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
                result[vix(t+eovol,g,s)] = (4.0 + mass) * vo[vix(t,g,s)];
            }
        }
        

        for (int mu = 0; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vix(t+eovol,g,s)] += (
                            std::conj(Ue[uix(hops_o[hix(t,mu,0)],mu,gi,g,eovol)])
                            * (
                                -ve[vix(hops_o[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * ve[vix(hops_o[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + Uo[uix(t,mu,g,gi,eovol)]
                            * (
                                -ve[vix(hops_o[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * ve[vix(hops_o[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }

    return result;
}



// separate matrices wilson dirac operator with runtime computed hops on even-odd grid
// the mass term has its own t loop
std::complex<double>* dw_eo_dir_mass_pmusggi (const std::complex<double>* Ue, const std::complex<double>* Uo,
                                 const std::complex<double>* ve, std::complex<double>* vo,
                                 double mass, int eodim[4]){

    // the even/odd lattices have the same dimensions as the full one, except that the t axis is halved


    // size of space-time, spin and gauge axes
    int v_size [6] = {eodim[0], eodim[1], eodim[2], eodim[3], 4, 3};

    // number of different fields and size of space-time and gauge axes
    int u_size [7] = {4, eodim[0], eodim[1], eodim[2], eodim[3], 3, 3};

    // size of the result tensor, which is even and odd stacked
    int r_size [7] = {2,eodim[0], eodim[1], eodim[2], eodim[3], 4, 3};

    // strides of the memory blocks
    int vstride [6];
    vstride[5] = 1;
    for (int sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    // strides of the memory blocks
    int ustride [7];
    ustride[6] = 1;
    for (int sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }
    // strides of the memory blocks
    int rstride [7];
    rstride[6] = 1;
    for (int sj = 5; sj >= 0; sj--){
        rstride[sj] = rstride[sj+1] * r_size[sj+1];
    }

    // the output is flattened in the space-time lattice
    std::complex<double>* result = new std::complex<double> [2*eodim[0]*eodim[1]*eodim[2]*eodim[3]*4*3];


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

    // mass term separate
#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){
                    // mass term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        
                            result[ptridx7(0,x,y,z,t,s,g,rstride)] = (4.0 + mass) * ve[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        int teo = x%2;
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){

                    // mu = 0 term
                    for (int s = 0; s < 4; s++){
                        for (int g = 0; g < 3; g++){
                            for (int gi = 0; gi < 3; gi++){
                                result[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -vo[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * vo[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + Ue[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * vo[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -vo[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * vo[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + Ue[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * vo[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -vo[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * vo[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + Ue[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * vo[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    // for this term, because the even and odd v are shrinked in t,
                    // we have to access different points than before:
                    // the first even and odd site in each t row have the same address on their grids
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(0,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Uo[ptridx7(3,x,y,z,(t-1+teo+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -vo[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * vo[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + Ue[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -vo[ptridx6(x,y,z,(t+teo)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * vo[ptridx6(x,y,z,(t+teo)%v_size[3],gamx[3][s],gi,vstride)]
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


    // now the odd term
    // teo again defined in the x loop

    // mass term separate
#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        
                            result[ptridx7(1,x,y,z,t,s,g,rstride)] = (4.0 + mass) * vo[ptridx6(x,y,z,t,s,g,vstride)];
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (int x = 0; x < v_size[0]; x++){
        int teo = (x+1)%2;
        for (int y = 0; y < v_size[1]; y++){
            for (int z = 0; z < v_size[2]; z++){
                for (int t = 0; t < v_size[3]; t++){

                    // mu = 0 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue[ptridx7(0,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                    * (
                                        -ve[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                        -gamf[0][s] * ve[ptridx6((x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                    + Uo[ptridx7(0,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve[ptridx6((x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                        +gamf[0][s] * ve[ptridx6((x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 1 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue[ptridx7(1,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                    * (
                                        -ve[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                        -gamf[1][s] * ve[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                    + Uo[ptridx7(1,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve[ptridx6(x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                        +gamf[1][s] * ve[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 2 term
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue[ptridx7(2,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                    * (
                                        -ve[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                        -gamf[2][s] * ve[ptridx6(x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                    + Uo[ptridx7(2,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve[ptridx6(x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                        +gamf[2][s] * ve[ptridx6(x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                    )
                                ) * 0.5;
                            }
                        }
                    }

                    // mu = 3 term
                    // for this term, because the even and odd v are shrinked in t,
                    // we have to access different points than before
                    // the first even and odd site in each t row have the same address on their grids
                    for (int s = 0; s < 4; s++){
                    for (int g = 0; g < 3; g++){
                        for (int gi = 0; gi < 3; gi++){
                            
                                result[ptridx7(1,x,y,z,t,s,g,rstride)] += (
                                    std::conj(Ue[ptridx7(3,x,y,z,(t-1+teo+u_size[4])%u_size[4],gi,g,ustride)])
                                    * (
                                        -ve[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],s,gi,vstride)]
                                        -gamf[3][s] * ve[ptridx6(x,y,z,(t-1+teo+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                    )
                                    + Uo[ptridx7(3,x,y,z,t,g,gi,ustride)]
                                    * (
                                        -ve[ptridx6(x,y,z,(t+teo)%v_size[3],s,gi,vstride)]
                                        +gamf[3][s] * ve[ptridx6(x,y,z,(t+teo)%v_size[3],gamx[3][s],gi,vstride)]
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



    return result;
}


// separate matrices wilson dirac operator with precomputed hops on even-odd grid
// the mass term has its own t loop
std::complex<double>* dw_eo_look_mass_pmusggi (const std::complex<double>* Ue, const std::complex<double>* Uo,
                                          const std::complex<double>* ve, const std::complex<double>* vo,
                                          const int* hops_e, const int* hops_o, double mass, int eovol){


    std::complex<double>* result = new std::complex<double> [2*eovol*4*3];

// even lattice part

#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        
        // mass term independent
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vix(t,g,s)] = (4.0 + mass) * ve[vix(t,g,s)];
            }
        }
    }

#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        for (int mu = 0; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vix(t,g,s)] += (
                            std::conj(Uo[uix(hops_e[hix(t,mu,0)],mu,gi,g,eovol)])
                            * (
                                -vo[vix(hops_e[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * vo[vix(hops_e[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + Ue[uix(t,mu,g,gi,eovol)]
                            * (
                                -vo[vix(hops_e[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * vo[vix(hops_e[hix(t,mu,1)],gi,gamx[mu][s])]
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
                result[vix(t+eovol,g,s)] = (4.0 + mass) * vo[vix(t,g,s)];
            }
        }
    }

#pragma omp parallel for
    for (int t = 0; t < eovol; t++){
        
        for (int mu = 0; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vix(t+eovol,g,s)] += (
                            std::conj(Ue[uix(hops_o[hix(t,mu,0)],mu,gi,g,eovol)])
                            * (
                                -ve[vix(hops_o[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * ve[vix(hops_o[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + Uo[uix(t,mu,g,gi,eovol)]
                            * (
                                -ve[vix(hops_o[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * ve[vix(hops_o[hix(t,mu,1)],gi,gamx[mu][s])]
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

    // switch between larger and smaller interval
    bool small_interval = true;

    int Lxyz[4] = {4,4,2,4};
    if (small_interval){
        Lxyz[0] = 10;
        Lxyz[1] = 10;
        Lxyz[2] = 8;
        Lxyz[3] = 32;
    }

    // add result to noopt to prevent compiler from optimizing out function calls
    std::complex<double> noopt (0.0,0.0);

    // random number generators
    std::mt19937 generator(15);
    std::normal_distribution<double> normal(0.0, 1.0);

    bool verbose_hops = false;

    // in each loop iteration, one lattice axis is doubled
    // or for the smaller interval, one axis is lengthened by 2
    for (int L_incr = 0; L_incr < 16; L_incr++){
        std::cout << "---" << std::endl;

        // change in the axis
        if (small_interval){
            Lxyz[(L_incr+2)%3] += 2;
        } else {
            Lxyz[(L_incr+2)%4] *= 2;
        }
        

        int Lx = Lxyz[0];
        int Ly = Lxyz[1];
        int Lz = Lxyz[2];
        int Lt = Lxyz[3];
        int vol = Lx * Ly * Lz * Lt;
        // strides of the grid
        int st [4] = {Ly*Lz*Lt, Lz*Lt, Lt, 1};
        // even/odd grid
        int Lteo = Lt/2;
        int Lxyzeo [4] = {Lx, Ly, Lz, Lteo};
        int steo [4] = {Ly*Lz*Lteo, Lz*Lteo, Lteo, 1};
        // constant variables do not change the compilation
        int n_gauge = 3;
        int n_spin = 4;
        

        int uvol = vol * 4 * n_gauge * n_gauge;
        int vvol = vol * n_gauge * n_spin;

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

        // for the even and odd fields, we use random numbers as pseudo fields
        // performance is not different if values are incorrect
        std::complex<double>* Ue = new std::complex<double>[uvol/2];
        for (int i = 0; i < uvol/2; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            Ue[i] = std::complex<double> (ree,imm);
        }
        std::complex<double>* Uo = new std::complex<double>[uvol/2];
        for (int i = 0; i < uvol/2; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            Uo[i] = std::complex<double> (ree,imm);
        }
        std::complex<double>* ve = new std::complex<double>[vvol/2];
        for (int i = 0; i < vvol/2; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            ve[i] = std::complex<double> (ree,imm);
        }
        std::complex<double>* vo = new std::complex<double>[vvol/2];
        for (int i = 0; i < vvol/2; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            vo[i] = std::complex<double> (ree,imm);
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

        // hops for the even and odd lattices
        // these are hops on the specified lattice that go onto the other lattice
        int * hops_e = new int [vol*4];
        for (int x = 0; x < Lx; x++){
            for (int y = 0; y < Ly; y++){
                for (int z = 0; z < Lz; z++){
                    int teo = (x+y+z)%2;
                    for (int t = 0; t < Lteo; t++){
                        hops_e[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8]
                            = (x-1+Lx)%Lx*steo[0]+y*steo[1]+z*steo[2]+t*steo[3];
                        hops_e[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+1]
                            = (x+1)%Lx*steo[0]+y*steo[1]+z*steo[2]+t*steo[3];
                        hops_e[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+2]
                            = x*steo[0]+(y-1+Ly)%Ly*steo[1]+z*steo[2]+t*steo[3];
                        hops_e[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+3]
                            = x*steo[0]+(y+1)%Ly*steo[1]+z*steo[2]+t*steo[3];
                        hops_e[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+4]
                            = x*steo[0]+y*steo[1]+(z-1+Lz)%Lz*steo[2]+t*steo[3];
                        hops_e[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+5]
                            = x*steo[0]+y*steo[1]+(z+1)%Lz*steo[2]+t*steo[3];
                        hops_e[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+6]
                            = x*steo[0]+y*steo[1]+z*steo[2]+(t-1+teo+Lteo)%Lteo*steo[3];
                        hops_e[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+7]
                            = x*steo[0]+y*steo[1]+z*steo[2]+(t+teo)%Lteo*steo[3];
                    }
                }
            }
        }
        int * hops_o = new int [vol*4];
        for (int x = 0; x < Lx; x++){
            for (int y = 0; y < Ly; y++){
                for (int z = 0; z < Lz; z++){
                    int teo = (x+y+z+1)%2;
                    for (int t = 0; t < Lteo; t++){
                        hops_o[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8]
                            = (x-1+Lx)%Lx*steo[0]+y*steo[1]+z*steo[2]+t*steo[3];
                        hops_o[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+1]
                            = (x+1)%Lx*steo[0]+y*steo[1]+z*steo[2]+t*steo[3];
                        hops_o[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+2]
                            = x*steo[0]+(y-1+Ly)%Ly*steo[1]+z*steo[2]+t*steo[3];
                        hops_o[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+3]
                            = x*steo[0]+(y+1)%Ly*steo[1]+z*steo[2]+t*steo[3];
                        hops_o[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+4]
                            = x*steo[0]+y*steo[1]+(z-1+Lz)%Lz*steo[2]+t*steo[3];
                        hops_o[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+5]
                            = x*steo[0]+y*steo[1]+(z+1)%Lz*steo[2]+t*steo[3];
                        hops_o[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+6]
                            = x*steo[0]+y*steo[1]+z*steo[2]+(t-1+teo+Lteo)%Lteo*steo[3];
                        hops_o[(x*steo[0]+y*steo[1]+z*steo[2]+t*steo[3])*8+7]
                            = x*steo[0]+y*steo[1]+z*steo[2]+(t+teo)%Lteo*steo[3];
                    }
                }
            }
        }

        std::cout << vol << " sites" << std::endl;
        std::cout << "grid layout " << Lx << " " << Ly << " " << Lz << " " << Lt << std::endl;
        std::cout << "even-odd grid: Lt = " << Lteo << std::endl;

        double * times_dir = new double [n_rep];
        double * times_look = new double [n_rep];
        double * times_dir_eo = new double [n_rep];
        double * times_look_eo = new double [n_rep];
        double * times_dir_mass_eo = new double [n_rep];
        double * times_look_mass_eo = new double [n_rep];

        for (int n = 0; n < n_warmup; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[32] = randval;

            std::complex<double> * result = dw_direct_musggi(U,v,Lxyz,mass);
            noopt += result[12];
            delete[] result;

            std::complex<double> * result2 = dw_look_musggi(U,v,hops,mass,vol);
            noopt += result2[23];
            delete[] result2;

            std::complex<double> * result3 = dw_eo_dir_pmusggi(Ue,Uo,ve,vo,mass,Lxyzeo);
            noopt += result3[92];
            
            std::complex<double> * result4 = dw_eo_look_pmusggi(Ue,Uo,ve,vo,hops_e,hops_o,mass,vol/2);
            noopt += result4[92];

            std::complex<double> * result5 = dw_eo_dir_mass_pmusggi(Ue,Uo,ve,vo,mass,Lxyzeo);
            noopt += result5[46];
            
            std::complex<double> * result6 = dw_eo_look_mass_pmusggi(Ue,Uo,ve,vo,hops_e,hops_o,mass,vol/2);
            noopt += result6[44];

            if (n == 0){
                int incorrect = 0;
                for (int i = 0; i < vvol; i++){
                    if (abs(result3[i]-result4[i]) > 0.001 || abs(result3[i]-result5[i]) > 0.001 || abs(result3[i]-result6[i]) > 0.001){
                        incorrect++;
                    }
                }
                std::cout << "no. of sites that are different for even-odd: " << incorrect << std::endl;
            }
            delete[] result3;
            delete[] result4;
            delete[] result5;
            delete[] result6;
        }

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[34] = randval;

            auto start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result = dw_direct_musggi(U,v,Lxyz,mass);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_dir[n] = double(dur.count());
            noopt += result[12];
            delete[] result;
            
            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result2 = dw_look_musggi(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_look[n] = double(dur.count());
            noopt += result2[23];
            delete[] result2;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result3 = dw_eo_dir_pmusggi(Ue,Uo,ve,vo,mass,Lxyzeo);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_dir_eo[n] = double(dur.count());
            noopt += result3[92];
            delete[] result3;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result4 = dw_eo_look_pmusggi(Ue,Uo,ve,vo,hops_e,hops_o,mass,vol/2);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_look_eo[n] = double(dur.count());
            noopt += result4[92];
            delete[] result4;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result5 = dw_eo_dir_mass_pmusggi(Ue,Uo,ve,vo,mass,Lxyzeo);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_dir_mass_eo[n] = double(dur.count());
            noopt += result5[4];
            delete[] result5;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result6 = dw_eo_look_mass_pmusggi(Ue,Uo,ve,vo,hops_e,hops_o,mass,vol/2);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_look_mass_eo[n] = double(dur.count());
            noopt += result6[3];
            delete[] result6;
        }

        delete [] U;
        delete [] v;
        delete [] Ue;
        delete [] Uo;
        delete [] ve;
        delete [] vo;
        delete [] hops;
        delete [] hops_e;
        delete [] hops_o;


        double avg_dir = 0.0;
        double avg_look = 0.0;
        double avg_dir_eo = 0.0;
        double avg_look_eo = 0.0;
        double avg_dir_mass_eo = 0.0;
        double avg_look_mass_eo = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_dir,times_dir+n_rep);
        std::sort(times_look,times_look+n_rep);
        std::sort(times_dir_eo,times_dir_eo+n_rep);
        std::sort(times_look_eo,times_look_eo+n_rep);
        std::sort(times_dir_mass_eo,times_dir_mass_eo+n_rep);
        std::sort(times_look_mass_eo,times_look_mass_eo+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_dir += times_dir[n];
            avg_look += times_look[n];
            avg_dir_eo += times_dir_eo[n];
            avg_look_eo += times_look_eo[n];
            avg_dir_mass_eo += times_dir_mass_eo[n];
            avg_look_mass_eo += times_look_mass_eo[n];
        }

        avg_dir /= double(n_res);
        avg_look /= double(n_res);
        avg_dir_eo /= double(n_res);
        avg_look_eo /= double(n_res);
        avg_dir_mass_eo /= double(n_res);
        avg_look_mass_eo /= double(n_res);

        std::cout << "runtime_index time: " << avg_dir/1000 << " µs\n";
        std::cout << "lookup_index time: " << avg_look/1000 << " µs\n";
        std::cout << "runtime_index_eo time: " << avg_dir_eo/1000 << " µs\n";
        std::cout << "lookup_index_eo time: " << avg_look_eo/1000 << " µs\n";
        std::cout << "runtime_index_eo_sep_mas time: " << avg_dir_mass_eo/1000 << " µs\n";
        std::cout << "lookup_index_eo_sep_mass time: " << avg_look_mass_eo/1000 << " µs\n";

        delete [] times_dir;
        delete [] times_look;
        delete [] times_dir_eo;
        delete [] times_look_eo;
        delete [] times_dir_mass_eo;
        delete [] times_look_mass_eo;
        
    }

    std::cout << "print noopt to prevent optimization: " << noopt << std::endl;
}

