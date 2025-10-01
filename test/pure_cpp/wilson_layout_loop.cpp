// test the separate matrix wilson dirac operator
// for different layouts and loop orderings
// with precomputed neighbour addresses

#include <omp.h>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <complex>



// address for complex auge field in layout U[t,mu,g,h]
inline int uix (int t, int mu, int g, int gi){
    return t*36 + mu*9 + g*3 + gi;
}
// address for complex fermion field in layout v[t,h,s]
inline int vix (int t, int g, int s){
    return t*12 + g*4 + s;
}
// address for hops in either layout
inline int hix (int t, int h, int d){
    return t*8 + h*2 + d;
}
// address for complex gauge field in layout U[mu,t,g,h]
inline int uixoc (int t, int mu, int g, int gi, int vol){
    return mu*vol*9 + t*9 + g*3 + gi;
}
// address for complex fermion field in layout v[t,s,h]
inline int vixoc (int t, int g, int s){
    return t*12 + s*3 + g;
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



std::complex<double>* dw_tmgs_muggis (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[t,mu,g,h] and v[t,h,s]

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
                            std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g)])
                            * (
                                -v[vix(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uix(t,mu,g,gi)]
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


std::complex<double>* dw_tmgs_mugsgi (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[t,mu,g,h] and v[t,h,s]

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
                for (int s = 0; s < 4; s++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vix(t,g,s)] += (
                            std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g)])
                            * (
                                -v[vix(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uix(t,mu,g,gi)]
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

std::complex<double>* dw_mtgs_muggis (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,h,s]

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
                            std::conj(U[uixoc(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vix(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uixoc(t,mu,g,gi,vol)]
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


std::complex<double>* dw_mtgs_mugsgi (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,h,s]

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
                for (int s = 0; s < 4; s++){
                    for (int gi = 0; gi < 3; gi++){
                        
                        result[vix(t,g,s)] += (
                            std::conj(U[uixoc(hops[hix(t,mu,0)],mu,gi,g,vol)])
                            * (
                                -v[vix(hops[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v[vix(hops[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U[uixoc(t,mu,g,gi,vol)]
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


std::complex<double>* dw_mtsg_mugsgi (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){
                result[vixoc(t,g,s)] = (4.0 + mass) * v[vixoc(t,g,s)];
            }
        }

        for (int mu = 0; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s++){
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


std::complex<double>* dw_mtsg_muggis (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h] and v[t,s,h]

    std::complex<double>* result = new std::complex<double> [vol*4*3];

#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s++){
                result[vixoc(t,g,s)] = (4.0 + mass) * v[vixoc(t,g,s)];
            }
        }

        for (int mu = 0; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int gi = 0; gi < 3; gi++){
                    for (int s = 0; s < 4; s++){
                    
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

        double * times_mtsg_muggis = new double [n_rep];
        double * times_mtsg_mugsgi = new double [n_rep];
        double * times_tmgs_muggis = new double [n_rep];
        double * times_tmgs_mugsgi = new double [n_rep];
        double * times_mtgs_muggis = new double [n_rep];
        double * times_mtgs_mugsgi = new double [n_rep];

        for (int n = 0; n < n_warmup; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[32] = randval;

            int incorrect = 0;

            std::complex<double> * result = dw_mtsg_muggis(U,v,hops,mass,vol);
            noopt += result[12];

            std::complex<double> * result2 = dw_mtsg_mugsgi(U,v,hops,mass,vol);
            noopt += result2[23];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result2[i]-result[i]) > 0.001) incorrect++;
            }
            delete[] result;
            delete[] result2;

            std::complex<double> * result3 = dw_tmgs_muggis(U,v,hops,mass,vol);
            noopt += result3[25];

            std::complex<double>* result4 = dw_tmgs_mugsgi(U,v,hops,mass,vol);
            noopt += result4[12];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result4[i]-result3[i]) > 0.001) incorrect++;
            }
            delete[] result3;
            delete[] result4;

            std::complex<double> * result5 = dw_mtgs_muggis(U,v,hops,mass,vol);
            noopt += result5[25];

            std::complex<double>* result6 = dw_mtgs_mugsgi(U,v,hops,mass,vol);
            noopt += result6[12];
            if (n == 0) for (int i = 0; i < vvol; i++){
                if (abs(result6[i]-result5[i]) > 0.001) incorrect++;
            }
            delete[] result5;
            delete[] result6;

            if (n == 0){
                std::cout << "no. of incorrect sites for new ops: " << incorrect << std::endl;
            }
        }

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[31] = randval;

            auto start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result = dw_mtsg_muggis(U,v,hops,mass,vol);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_mtsg_muggis[n] = double(dur.count());
            noopt += result[12];
            delete[] result;
            
            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result2 = dw_mtsg_mugsgi(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_mtsg_mugsgi[n] = double(dur.count());
            noopt += result2[23];
            delete[] result2;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result3 = dw_tmgs_muggis(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_tmgs_muggis[n] = double(dur.count());
            noopt += result3[92];
            delete[] result3;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result4 = dw_tmgs_mugsgi(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_tmgs_mugsgi[n] = double(dur.count());
            noopt += result4[92];
            delete[] result4;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result5 = dw_mtgs_muggis(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_mtgs_muggis[n] = double(dur.count());
            noopt += result5[92];
            delete[] result5;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result6 = dw_mtgs_mugsgi(U,v,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_mtgs_mugsgi[n] = double(dur.count());
            noopt += result6[92];
            delete[] result6;

        }

        delete [] U;
        delete [] v;
        delete [] hops;

        double avg_mtsg_muggis = 0.0;
        double avg_mtsg_mugsgi = 0.0;
        double avg_tmgs_muggis = 0.0;
        double avg_tmgs_mugsgi = 0.0;
        double avg_mtgs_muggis = 0.0;
        double avg_mtgs_mugsgi = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_mtsg_muggis,times_mtsg_muggis+n_rep);
        std::sort(times_mtsg_mugsgi,times_mtsg_mugsgi+n_rep);
        std::sort(times_tmgs_muggis,times_tmgs_muggis+n_rep);
        std::sort(times_tmgs_mugsgi,times_tmgs_mugsgi+n_rep);
        std::sort(times_mtgs_muggis,times_mtgs_muggis+n_rep);
        std::sort(times_mtgs_mugsgi,times_mtgs_mugsgi+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_mtsg_muggis += times_mtsg_muggis[n];
            avg_mtsg_mugsgi += times_mtsg_mugsgi[n];
            avg_tmgs_muggis += times_tmgs_muggis[n];
            avg_tmgs_mugsgi += times_tmgs_mugsgi[n];
            avg_mtgs_muggis += times_mtgs_muggis[n];
            avg_mtgs_mugsgi += times_mtgs_mugsgi[n];
        }

        avg_mtsg_muggis /= double(n_res);
        avg_mtsg_mugsgi /= double(n_res);
        avg_tmgs_muggis /= double(n_res);
        avg_tmgs_mugsgi /= double(n_res);
        avg_mtgs_muggis /= double(n_res);
        avg_mtgs_mugsgi /= double(n_res);

        std::cout << "mtsg_muggis time: " << avg_mtsg_muggis/1000 << " µs\n";
        std::cout << "mtsg_mugsgi time: " << avg_mtsg_mugsgi/1000 << " µs\n";
        std::cout << "tmgs_muggis time: " << avg_tmgs_muggis/1000 << " µs\n";
        std::cout << "tmgs_mugsgi time: " << avg_tmgs_mugsgi/1000 << " µs\n";
        std::cout << "mtgs_muggis time: " << avg_mtgs_muggis/1000 << " µs\n";
        std::cout << "mtgs_mugsgi time: " << avg_mtgs_mugsgi/1000 << " µs\n";

        delete [] times_mtsg_muggis;
        delete [] times_mtsg_mugsgi;
        delete [] times_tmgs_muggis;
        delete [] times_tmgs_mugsgi;
        delete [] times_mtgs_muggis;
        delete [] times_mtgs_mugsgi;
        
    }

    std::cout << "print noopt to prevent optimization: " << noopt << std::endl;
}

