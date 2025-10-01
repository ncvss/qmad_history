// test wilson dirac operator with separate matrices and precomputed neighbour addresses
// try out different partitionings of the lattice
// for the domains that are parallelised and given to one thread
// has to be run on a machine with 16 threads

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




// the outermost loop is split
// this corresponds to the lattice being sliced perpendicular to the x-axis
std::complex<double>* dw_x_slice (const std::complex<double>* U,
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


// create blocks by halving the axes
// only works if axes are multiples of 2
std::complex<double>* dw_cube_blocks (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol, int Lx, int Ly, int Lz, int Lt){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for collapse(4)
    for (int xblock = 0; xblock < Lx; xblock += Lx/2){
        for (int yblock = 0; yblock < Ly; yblock += Ly/2){
            for (int zblock = 0; zblock < Lz; zblock += Lz/2){
                for (int tblock = 0; tblock < Lt; tblock += Lt/2){
                    for (int x = xblock; x < xblock+Lx/2; x++){
                        for (int y = yblock; y < yblock+Ly/2; y++){
                            for (int z = zblock; z < zblock+Lz/2; z++){
                                for (int tout = tblock; tout < tblock+Lt/2; tout++){
                                    int t = x*Ly*Lz*Lt + y*Lz*Lt + z*Lt + tout;

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
                            }
                        }
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
    int Lxyz[4] = {4,4,4,4};

    // random number generators
    std::mt19937 generator(16);
    std::normal_distribution<double> normal(0.0, 1.0);

    // add result to noopt to prevent compiler from optimizing out function calls
    std::complex<double> noopt (0.0,0.0);

    // in each loop iteration, one lattice axis is doubled
    for (int L_incr = 0; L_incr < 11; L_incr++){
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


        std::cout << vol << " sites" << std::endl;
        std::cout << "grid layout " << Lx << " " << Ly << " " << Lz << " " << Lt << std::endl;

        double * times_x = new double [n_rep];
        double * times_cube = new double [n_rep];

        for (int n = 0; n < n_warmup; n++){
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[3] = randval;
            
            std::complex<double> * result2 = dw_x_slice(U,v,hops,mass,vol);
            noopt += result2[13];
            if (n > 0){
                delete[] result2;
            }

            std::complex<double> * result3 = dw_cube_blocks(U,v,hops,mass,vol,Lx,Ly,Lz,Lt);
            noopt += result3[8];
            if (n == 0){
                int incorrect = 0;
                for (int cr = 0; cr < vvol; cr++){
                    if (abs(result2[cr]-result3[cr]) > 0.0001){
                        incorrect++;
                    }
                }
                std::cout << "number of incorrect sites: " << incorrect << std::endl;
                delete[] result2;
            }
            delete[] result3;
        }

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[33] = randval;
            
            auto start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result2 = dw_x_slice(U,v,hops,mass,vol);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_x[n] = double(dur.count());
            noopt += result2[13];
            delete[] result2;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result3 = dw_cube_blocks(U,v,hops,mass,vol,Lx,Ly,Lz,Lt);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_cube[n] = double(dur.count());
            noopt += result3[8];
            delete[] result3;
        }

        delete [] U;
        delete [] v;
        delete [] hops;


        double avg_x = 0.0;
        double avg_cube = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_x,times_x+n_rep);
        std::sort(times_cube,times_cube+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_x += times_x[n];
            avg_cube += times_cube[n];
        }

        avg_x /= double(n_res);
        avg_cube /= double(n_res);

        std::cout << "x-slice parallel domain time: " << avg_x/1000 << " µs\n";
        std::cout << "hypercube parallel domain time: " << avg_cube/1000 << " µs\n";

        delete [] times_x;
        delete [] times_cube;
    }

    std::cout << "print noopt to prevent optimization: " << noopt << std::endl;
}
