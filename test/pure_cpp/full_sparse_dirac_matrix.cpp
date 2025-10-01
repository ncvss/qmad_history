// make a sparse matrix multiplication that has nonzero entries at the same points as the dirac operator
// we use random numbers as entries, so the computation is not correct
// the performance is the same
// then compare with the best current non-simd implementation

#include <complex>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>


// address for complex gauge field in layout U[mu,t,g,h]
inline int uixoc (int t, int mu, int g, int gi, int vol){
    return mu*vol*9 + t*9 + g*3 + gi;
}
// address for complex fermion field in layout v[t,s,h]
inline int vixoc (int t, int g, int s){
    return t*12 + s*3 + g;
}
// address for hops
inline __attribute__((always_inline)) int hix (int t, int h, int d){
    return t*8 + h*2 + d;
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

std::complex<double>* dw_call_look_mass_musggi_om (const std::complex<double>* U,
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
                result[vixoc(t,g,s)] = (4.0 + mass) * v[vixoc(t,g,s)];

                for (int gi = 0; gi < 3; gi++){
                    
                    result[vixoc(t,g,s)] += (
                        std::conj(U[uixoc(hops[hix(t,mu,0)],mu,gi,g,vol)])
                        * (
                            -v[vixoc(hops[hix(t,mu,0)],gi,s)]
                            -gamf[mu][s] * v[vixoc(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                        )
                        + U[uixoc(t,mu,g,gi,vol)]
                        * (
                            -v[vixoc(hops[hix(t,mu,1)],gi,s)]
                            +gamf[mu][s] * v[vixoc(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                        )
                    ) * 0.5;
                }
            }
        }

        for (mu = 1; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixoc(t,g,s)] += (
                            std::conj(U[uixoc(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vixoc(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vixoc(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uixoc(t,mu,g,gi,vol)]
                            * (
                                -v[vixoc(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vixoc(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }


    return result;
}


// simple sparse matrix-vector multiplication
std::complex<double>* dw_call_fake_sparse (const std::complex<double>* vals,
                                          const std::complex<double>* v, int * comp_ptr, int vol){

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for
    for (int i = 0; i < vol*4*3; i++){
        std::complex<double> comp_res = 0;
        for (int j = 0; j < 49; j++){
            comp_res += vals[i*49+j]*v[comp_ptr[i*49+j]];
        }
        result[i] = comp_res;
    }

    return result;
}

std::complex<double>* dw_call_look_mass_musggi_om_nopar (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

    for (int t = 0; t < vol; t++){
        
        // mass term and mu = 0 term in same loop to minimize result access
        int mu = 0;
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vixoc(t,g,s)] = (4.0 + mass) * v[vixoc(t,g,s)];

                for (int gi = 0; gi < 3; gi++){
                    
                    result[vixoc(t,g,s)] += (
                        std::conj(U[uixoc(hops[hix(t,mu,0)],mu,gi,g,vol)])
                        * (
                            -v[vixoc(hops[hix(t,mu,0)],gi,s)]
                            -gamf[mu][s] * v[vixoc(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                        )
                        + U[uixoc(t,mu,g,gi,vol)]
                        * (
                            -v[vixoc(hops[hix(t,mu,1)],gi,s)]
                            +gamf[mu][s] * v[vixoc(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                        )
                    ) * 0.5;
                }
            }
        }

        for (mu = 1; mu < 4; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vixoc(t,g,s)] += (
                            std::conj(U[uixoc(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vixoc(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vixoc(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uixoc(t,mu,g,gi,vol)]
                            * (
                                -v[vixoc(hops[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v[vixoc(hops[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }


    return result;
}


// simple sparse mat-vec mul
std::complex<double>* dw_call_fake_sparse_nopar (const std::complex<double>* vals,
                                          const std::complex<double>* v, int * comp_ptr, int vol){

    std::complex<double>* result = new std::complex<double> [vol*4*3];

    for (int i = 0; i < vol*4*3; i++){
        std::complex<double> comp_res = 0;
        for (int j = 0; j < 49; j++){
            comp_res += vals[i*49+j]*v[comp_ptr[i*49+j]];
        }
        result[i] = comp_res;
    }

    return result;
}



int main (){
    int n_rep = 200;
    int n_warmup = 20;
    double mass = -0.5;
    std::cout << n_rep << " repetitions" << std::endl;
    std::cout << "mass = " << mass << std::endl;
    int Lxyz[4] = {4,4,4,4};

    // add result to noopt to prevent compiler from optimizing out function calls
    std::complex<double> noopt (0.0,0.0);

    // in each loop iteration, one lattice axis is doubled
    for (int L_incr = 0; L_incr < 12; L_incr++){
        std::cout << "---" << std::endl;
        int Lx = Lxyz[0];
        int Ly = Lxyz[1];
        int Lz = Lxyz[2];
        int Lt = Lxyz[3];
        Lxyz[(L_incr+3)%4] *= 2;
        int vol = Lx * Ly * Lz * Lt;
        // strides of the grid
        int st [4] = {Ly*Lz*Lt, Lz*Lt, Lt, 1};
        int n_gauge = 3;
        int n_spin = 4;
        // random number generators
        std::mt19937 generator(11);
        std::normal_distribution<double> normal(0.0, 1.0);

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

        // sparse matrix multiplication
        // vals has the data (random in this case)
        // comp_ptr has the input vector components that sum up to this output component
        // the input vector components for (t,s,g) are (t,s,g) and (t+mu,gamx[mu][s],all g)
        int* comp_ptr = new int [vvol*49];
        for (int t = 0; t < vol; t++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    // address that corresponds to t,s,g in the component lookup table
                    int currcomp = (t*12+s*3+g)*49;

                    // mass term
                    comp_ptr[currcomp] = t*12+s*3+g;

                    // hop terms: iterate over mu, sign of mu, and all 3 gi
                    // there are 2 hops: with and without gamma
                    for (int mu = 0; mu < 4; mu++){
                        for (int dir = 0; dir < 2; dir++){
                            for (int gi = 0; gi < 3; gi++){
                                comp_ptr[currcomp+1+mu*12+dir*6+gi] = hops[hix(t,mu,dir)]*12+s*3+gi;
                                comp_ptr[currcomp+1+mu*12+dir*6+gi+3] = hops[hix(t,mu,dir)]*12+gamx[mu][s]*3+gi;
                            }
                        }
                    }
                }
            }
        }


        // random complex numbers as the non-zero values from the wilson dirac matrix
        std::complex<double>* vals = new std::complex<double>[vvol*49];
        for (int i = 0; i < vvol*49; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            vals[i] = std::complex<double> (ree,imm);
        }

        std::cout << vol << " sites" << std::endl;
        std::cout << "grid layout " << Lx << " " << Ly << " " << Lz << " " << Lt << std::endl;

        double * times_stand = new double [n_rep];
        double * times_sparse = new double [n_rep];
        double * times_stand_nop = new double [n_rep];
        double * times_sparse_nop = new double [n_rep];

        for (int n = 0; n < n_warmup; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[34] = randval;

            std::complex<double> * result = dw_call_look_mass_musggi_om(U,v,hops,mass,vol);
            noopt += result[12];
            delete[] result;

            std::complex<double> * result2 = dw_call_fake_sparse(vals,v,comp_ptr,vol);
            noopt += result2[45];
            delete[] result2;

            std::complex<double> * result3 = dw_call_look_mass_musggi_om_nopar(U,v,hops,mass,vol);
            noopt += result3[2];
            delete[] result3;

            std::complex<double> * result4 = dw_call_fake_sparse_nopar(vals,v,comp_ptr,vol);
            noopt += result4[42];
            delete[] result4;
        }

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[34] = randval;

            auto start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result = dw_call_look_mass_musggi_om(U,v,hops,mass,vol);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_stand[n] = double(dur.count());
            noopt += result[17];
            delete[] result;
            
            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result2 = dw_call_fake_sparse(vals,v,comp_ptr,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_sparse[n] = double(dur.count());
            noopt += result2[28];
            delete[] result2;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result3 = dw_call_look_mass_musggi_om_nopar(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_stand_nop[n] = double(dur.count());
            noopt += result3[17];
            delete[] result3;
            
            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result4 = dw_call_fake_sparse_nopar(vals,v,comp_ptr,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_sparse_nop[n] = double(dur.count());
            noopt += result4[28];
            delete[] result4;
        }

        delete [] U;
        delete [] v;
        delete [] hops;
        delete [] vals;
        delete [] comp_ptr;


        double avg_stand = 0.0;
        double avg_sparse = 0.0;
        double avg_stand_nop = 0.0;
        double avg_sparse_nop = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_stand,times_stand+n_rep);
        std::sort(times_sparse,times_sparse+n_rep);
        std::sort(times_stand_nop,times_stand_nop+n_rep);
        std::sort(times_sparse_nop,times_sparse_nop+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_stand += times_stand[n];
            avg_sparse += times_sparse[n];
            avg_stand_nop += times_stand_nop[n];
            avg_sparse_nop += times_sparse_nop[n];
        }

        avg_stand /= double(n_res);
        avg_sparse /= double(n_res);
        avg_stand_nop /= double(n_res);
        avg_sparse_nop /= double(n_res);

        std::cout << "parallel_standard time: " << avg_stand/1000 << " µs\n";
        std::cout << "parallel_sparse matvec mul time: " << avg_sparse/1000 << " µs\n";
        std::cout << "noparallel_standard time: " << avg_stand_nop/1000 << " µs\n";
        std::cout << "noparallel_sparse matvec mul time: " << avg_sparse_nop/1000 << " µs\n";

        delete [] times_sparse;
        delete [] times_stand;
        delete [] times_sparse_nop;
        delete [] times_stand_nop;
    }

    std::cout << "print noopt to prevent optimization: " << noopt << std::endl;
}

