// show the advantage gained from unrolling the loop and putting the mass term in one iteration
// in the wilson dirac operator

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


// mu loop is unrolled and mass is inserted in one iteration
std::complex<double>* dw_musggi_mass_unroll (const std::complex<double>* U,
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


// mass term has separate s and g iteration
std::complex<double>* dw_musggi_mass_sep (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for
    for (int t = 0; t < vol; t++){
        
        // mass term and mu terms in separate loops
        for (int s = 0; s < 4; s++){
            for (int g = 0; g < 3; g++){
                result[vixoc(t,g,s)] = (4.0 + mass) * v[vixoc(t,g,s)];
            }
        }

        for (int mu = 0; mu < 4; mu++){
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
        // constant variables do not change the compilation
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

        std::cout << vol << " sites" << std::endl;
        std::cout << "grid layout " << Lx << " " << Ly << " " << Lz << " " << Lt << std::endl;

        double * times_unroll = new double [n_rep];
        double * times_sep = new double [n_rep];

        for (int n = 0; n < n_warmup; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[34] = randval;

            std::complex<double> * result = dw_musggi_mass_unroll(U,v,hops,mass,vol);
            noopt += result[12];

            std::complex<double> * result2 = dw_musggi_mass_sep(U,v,hops,mass,vol);
            noopt += result2[45];

            if (n == 0){
                int incorrect = 0;
                for (int i = 0; i < vvol; i++){
                    if (abs(result[i]-result2[i]) > 0.0001){
                        incorrect++;
                    }
                }
                std::cout << "no of incorrect points: " << incorrect << std::endl;
            }

            delete[] result;
            delete[] result2;
        }

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[34] = randval;

            auto start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result = dw_musggi_mass_unroll(U,v,hops,mass,vol);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_unroll[n] = double(dur.count());
            noopt += result[17];
            delete[] result;
            
            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result2 = dw_musggi_mass_sep(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_sep[n] = double(dur.count());
            noopt += result2[28];
            delete[] result2;

        }

        delete [] U;
        delete [] v;
        delete [] hops;


        double avg_unroll = 0.0;
        double avg_sep = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_unroll,times_unroll+n_rep);
        std::sort(times_sep,times_sep+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_unroll += times_unroll[n];
            avg_sep += times_sep[n];
        }

        avg_unroll /= double(n_res);
        avg_sep /= double(n_res);

        std::cout << "unrolled loop with mass term time: " << avg_unroll/1000 << " µs\n";
        std::cout << "separate mass term time: " << avg_sep/1000 << " µs\n";

        delete [] times_sep;
        delete [] times_unroll;
    }

    std::cout << "number that can be ignored: " << noopt << std::endl;
}

