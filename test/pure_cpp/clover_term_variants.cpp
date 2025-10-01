// test the different variants of the clover term:
// - runtime computation of the entire term
// - precomputation of F
// - precomputation of sigma x F
// we use random numbers instead of gauge fields, so computation is wrong
// the performance is the same

#include <omp.h>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <complex>


// sigx[munu][i] is the spin component of v that is proportional to spin component i of sigmamunu @ v
static const int sigx [6][4] =
    {{0,1,2,3},
     {1,0,3,2},
     {1,0,3,2},
     {1,0,3,2},
     {1,0,3,2},
     {0,1,2,3} };

// sigf[munu][i] is the prefactor of spin component i of sigmamunu @ v
static const std::complex<double> sigf [6][4] =
    {{std::complex<double>(0, 1), std::complex<double>(0,-1), std::complex<double>(0, 1), std::complex<double>(0,-1)},
     {   1,  -1,   1,  -1},
     {std::complex<double>(0, 1), std::complex<double>(0, 1), std::complex<double>(0, 1), std::complex<double>(0, 1)},
     {std::complex<double>(0,-1), std::complex<double>(0,-1), std::complex<double>(0, 1), std::complex<double>(0, 1)},
     {   1,  -1,  -1,   1},
     {std::complex<double>(0,-1), std::complex<double>(0, 1), std::complex<double>(0, 1), std::complex<double>(0,-1)} };


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

// address for field strength tensors (index order F[t,triangle index,block number])
// the upper triangle is flattened with the following indices:
//  0 | 1 | 2 | 3 | 4 | 5
//  1 | 6 | 7 | 8 | 9 |10
//  2 | 7 |11 |12 |13 |14
//  3 | 8 |12 |15 |16 |17
//  4 | 9 |13 |16 |18 |19
//  5 |10 |14 |17 |19 |20
// (the lower triangles are the same numbers, but conjugated)
inline int sfix (int t, int sblock, int triix){
    return t*42 + triix*2 + sblock;
}
// address for field strength tensor F[munu,t,g,gi]
inline int fix (int munu, int t, int g, int gi, int vol){
    return munu*vol*9 + t*9 + g*3 + gi;
}
// address for field strength tensor F[t,munu,g,gi]
inline int fix2 (int t,int munu,  int g, int gi){
    return t*36 + munu*9 + g*3 + gi;
}


// directly compute the entire field strength term at runtime
// use the component permutation version of sigma
inline void clover_direct (const std::complex<double> *U, const std::complex<double> *v, std::complex<double> *result, const int * hops,
                    int vol, double csw, int t){

    // term for Q_munu
    int munu = 0;
    for (int mu = 0; mu < 4; mu++){
        for (int nu = 0; nu < mu; nu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        for (int gj = 0; gj < 3; gj++){
                            for (int gk = 0; gk < 3; gk++){
                                for (int gl = 0; gl < 3; gl++){
                                    result[vix(t,g,s)] +=
                                        -csw * 0.0625 * (
                                            U[uix(t,mu,g,gi,vol)]
                                            * U[uix(hops[hix(t,mu,1)],nu,gi,gj,vol)]
                                            * std::conj(U[uix(hops[hix(t,nu,1)],mu,gk,gj,vol)])
                                            * std::conj(U[uix(t,nu,gl,gk,vol)])
                                            + U[uix(t,nu,g,gi,vol)]
                                            * std::conj(U[uix(hops[hix(hops[hix(t,mu,0)],nu,1)],mu,gj,gi,vol)])
                                            * std::conj(U[uix(hops[hix(t,mu,0)],nu,gk,gj,vol)])
                                            * U[uix(hops[hix(t,mu,0)],mu,gk,gl,vol)]
                                            + std::conj(U[uix(hops[hix(t,nu,0)],nu,gi,g,vol)])
                                            * U[uix(hops[hix(t,nu,0)],mu,gi,gj,vol)]
                                            * U[uix(hops[hix(hops[hix(t,mu,1)],nu,0)],nu,gj,gk,vol)]
                                            * std::conj(U[uix(t,mu,gl,gk,vol)])
                                            + std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)])
                                            * std::conj(U[uix(hops[hix(hops[hix(t,mu,0)],nu,0)],nu,gj,gi,vol)])
                                            * U[uix(hops[hix(hops[hix(t,mu,0)],nu,0)],mu,gj,gk,vol)]
                                            * U[uix(hops[hix(t,nu,0)],nu,gk,gl,vol)]
                                        ) * sigf[munu][s] * v[vix(t,gl,sigx[munu][s])];
                                }
                            }
                        }
                    }
                }
            }
            munu++;
        }
    }

    // term for Q_numu
    munu = 0;
    for (int nu = 0; nu < 4; nu++){
        for (int mu = 0; mu < nu; mu++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        for (int gj = 0; gj < 3; gj++){
                            for (int gk = 0; gk < 3; gk++){
                                for (int gl = 0; gl < 3; gl++){
                                    result[vix(t,g,s)] +=
                                        csw * 0.0625 * (
                                            U[uix(t,mu,g,gi,vol)]
                                            * U[uix(hops[hix(t,mu,1)],nu,gi,gj,vol)]
                                            * std::conj(U[uix(hops[hix(t,nu,1)],mu,gk,gj,vol)])
                                            * std::conj(U[uix(t,nu,gl,gk,vol)])
                                            + U[uix(t,nu,g,gi,vol)]
                                            * std::conj(U[uix(hops[hix(hops[hix(t,mu,0)],nu,1)],mu,gj,gi,vol)])
                                            * std::conj(U[uix(hops[hix(t,mu,0)],nu,gk,gj,vol)])
                                            * U[uix(hops[hix(t,mu,0)],mu,gk,gl,vol)]
                                            + std::conj(U[uix(hops[hix(t,nu,0)],nu,gi,g,vol)])
                                            * U[uix(hops[hix(t,nu,0)],mu,gi,gj,vol)]
                                            * U[uix(hops[hix(hops[hix(t,mu,1)],nu,0)],nu,gj,gk,vol)]
                                            * std::conj(U[uix(t,mu,gl,gk,vol)])
                                            + std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)])
                                            * std::conj(U[uix(hops[hix(hops[hix(t,mu,0)],nu,0)],nu,gj,gi,vol)])
                                            * U[uix(hops[hix(hops[hix(t,mu,0)],nu,0)],mu,gj,gk,vol)]
                                            * U[uix(hops[hix(t,nu,0)],nu,gk,gl,vol)]
                                        ) * sigf[munu][s] * v[vix(t,gl,sigx[munu][s])];
                                }
                            }
                        }
                    }
                }
            }
            munu++;
        }
    }
    
}


// rearrange the clover term using the distributive law
// we multiply the rightmost matrix onto the colour vector, then the next one to the left, and so on
inline void clover_direct_rearr (const std::complex<double> *U, const std::complex<double> *v, std::complex<double> *result, const int * hops,
                    int vol, double csw, int t){

    // term for Q_munu
    int munu = 0;
    for (int mu = 0; mu < 4; mu++){
        for (int nu = 0; nu < mu; nu++){
            for (int s = 0; s < 4; s++){
                // initialise with the vector
                std::complex<double> aggr[3];
                for (int gl = 0; gl < 3; gl++){
                    aggr[gl] = -csw * 0.0625 * sigf[munu][s] * v[vix(t,gl,sigx[munu][s])];
                }
                
                // multiply last matrix from last term
                std::complex<double> aggr2[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr2[gk] += U[uix(hops[hix(t,nu,0)],nu,gk,gl,vol)] * aggr[gl];
                    }
                }
                // next matrix
                std::complex<double> aggr3[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr3[gj] += U[uix(hops[hix(hops[hix(t,mu,0)],nu,0)],mu,gj,gk,vol)] * aggr2[gk];
                    }
                }
                std::complex<double> aggr4[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr4[gi] += std::conj(U[uix(hops[hix(hops[hix(t,mu,0)],nu,0)],nu,gj,gi,vol)]) * aggr3[gj];
                    }
                }
                // we can add the final contribution to the result
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vix(t,g,s)] += std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)]) * aggr4[gi];
                    }
                }

                // now the same for the other 3 terms

                std::complex<double> aggr5[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr5[gk] += std::conj(U[uix(t,mu,gl,gk,vol)]) * aggr[gl];
                    }
                }
                std::complex<double> aggr6[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr6[gj] += U[uix(hops[hix(hops[hix(t,mu,1)],nu,0)],nu,gj,gk,vol)] * aggr5[gk];
                    }
                }
                std::complex<double> aggr7[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr7[gi] += U[uix(hops[hix(t,nu,0)],mu,gi,gj,vol)] * aggr6[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vix(t,g,s)] += std::conj(U[uix(hops[hix(t,nu,0)],nu,gi,g,vol)]) * aggr7[gi];
                    }
                }

                std::complex<double> aggr8[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr8[gk] += U[uix(hops[hix(t,mu,0)],mu,gk,gl,vol)] * aggr[gl];
                    }
                }
                std::complex<double> aggr9[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr9[gj] += std::conj(U[uix(hops[hix(t,mu,0)],nu,gk,gj,vol)]) * aggr8[gk];
                    }
                }
                std::complex<double> aggr10[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr10[gi] += std::conj(U[uix(hops[hix(hops[hix(t,mu,0)],nu,1)],mu,gj,gi,vol)]) * aggr9[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vix(t,g,s)] += U[uix(t,nu,g,gi,vol)] * aggr10[gi];
                    }
                }

                std::complex<double> aggr11[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr11[gk] += std::conj(U[uix(t,nu,gl,gk,vol)]) * aggr[gl];
                    }
                }
                std::complex<double> aggr12[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr12[gj] += std::conj(U[uix(hops[hix(t,nu,1)],mu,gk,gj,vol)]) * aggr11[gk];
                    }
                }
                std::complex<double> aggr13[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr13[gi] += U[uix(hops[hix(t,mu,1)],nu,gi,gj,vol)] * aggr12[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vix(t,g,s)] += U[uix(t,mu,g,gi,vol)] * aggr13[gi];
                    }
                }
            }
            munu++;
        }
    }

    // term for Q_numu
    munu = 0;
    for (int nu = 0; nu < 4; nu++){
        for (int mu = 0; mu < nu; mu++){
            for (int s = 0; s < 4; s++){
                // initialise with the vector
                std::complex<double> aggr[3];
                for (int gl = 0; gl < 3; gl++){
                    aggr[gl] = -csw * 0.0625 * sigf[munu][s] * v[vix(t,gl,sigx[munu][s])];
                }
                
                // multiply last matrix from last term
                std::complex<double> aggr2[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr2[gk] += U[uix(hops[hix(t,nu,0)],nu,gk,gl,vol)] * aggr[gl];
                    }
                }
                // next matrix
                std::complex<double> aggr3[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr3[gj] += U[uix(hops[hix(hops[hix(t,mu,0)],nu,0)],mu,gj,gk,vol)] * aggr2[gk];
                    }
                }
                std::complex<double> aggr4[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr4[gi] += std::conj(U[uix(hops[hix(hops[hix(t,mu,0)],nu,0)],nu,gj,gi,vol)]) * aggr3[gj];
                    }
                }
                // we can add the final contribution to the result
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vix(t,g,s)] += -std::conj(U[uix(hops[hix(t,mu,0)],mu,gi,g,vol)]) * aggr4[gi];
                    }
                }

                // now the same for the other 3 terms

                std::complex<double> aggr5[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr5[gk] += std::conj(U[uix(t,mu,gl,gk,vol)]) * aggr[gl];
                    }
                }
                std::complex<double> aggr6[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr6[gj] += U[uix(hops[hix(hops[hix(t,mu,1)],nu,0)],nu,gj,gk,vol)] * aggr5[gk];
                    }
                }
                std::complex<double> aggr7[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr7[gi] += U[uix(hops[hix(t,nu,0)],mu,gi,gj,vol)] * aggr6[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vix(t,g,s)] += -std::conj(U[uix(hops[hix(t,nu,0)],nu,gi,g,vol)]) * aggr7[gi];
                    }
                }

                std::complex<double> aggr8[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr8[gk] += U[uix(hops[hix(t,mu,0)],mu,gk,gl,vol)] * aggr[gl];
                    }
                }
                std::complex<double> aggr9[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr9[gj] += std::conj(U[uix(hops[hix(t,mu,0)],nu,gk,gj,vol)]) * aggr8[gk];
                    }
                }
                std::complex<double> aggr10[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr10[gi] += std::conj(U[uix(hops[hix(hops[hix(t,mu,0)],nu,1)],mu,gj,gi,vol)]) * aggr9[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vix(t,g,s)] += -U[uix(t,nu,g,gi,vol)] * aggr10[gi];
                    }
                }

                std::complex<double> aggr11[3] = {0};
                for (int gk = 0; gk < 3; gk++){
                    for (int gl = 0; gl < 3; gl++){
                        aggr11[gk] += std::conj(U[uix(t,nu,gl,gk,vol)]) * aggr[gl];
                    }
                }
                std::complex<double> aggr12[3] = {0};
                for (int gj = 0; gj < 3; gj++){
                    for (int gk = 0; gk < 3; gk++){
                        aggr12[gj] += std::conj(U[uix(hops[hix(t,nu,1)],mu,gk,gj,vol)]) * aggr11[gk];
                    }
                }
                std::complex<double> aggr13[3] = {0};
                for (int gi = 0; gi < 3; gi++){
                    for (int gj = 0; gj < 3; gj++){
                        aggr13[gi] += U[uix(hops[hix(t,mu,1)],nu,gi,gj,vol)] * aggr12[gj];
                    }
                }
                for (int g = 0; g < 3; g++){
                    for (int gi = 0; gi < 3; gi++){
                        result[vix(t,g,s)] += -U[uix(t,mu,g,gi,vol)] * aggr13[gi];
                    }
                }
            }
            munu++;
        }
    }
    
}


// clover term that takes a precomputed sigma F tensor product
inline void clover_grid (const std::complex<double> *v, std::complex<double> *result, const std::complex<double> *sF,
                  int vol, int t){
    
    // iterate over the 2 triangles which correspond to spin 0,1 and 2,3 respectively
    // memory layout is sigmaF[t,triangle index,block number]
    for (int sbl = 0; sbl < 2; sbl++){
        // contribution from s=0,g=0
        std::complex<double> v00 = v[vix(t,0,sbl*2)];
        result[vix(t,0,sbl*2)] += sF[sfix(t,sbl,0)]*v00;
        result[vix(t,1,sbl*2)] += std::conj(sF[sfix(t,sbl,1)])*v00;
        result[vix(t,2,sbl*2)] += std::conj(sF[sfix(t,sbl,2)])*v00;
        result[vix(t,0,sbl*2+1)] += std::conj(sF[sfix(t,sbl,3)])*v00;
        result[vix(t,1,sbl*2+1)] += std::conj(sF[sfix(t,sbl,4)])*v00;
        result[vix(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,5)])*v00;

        // contribution from s=0,g=1
        std::complex<double> v01 = v[vix(t,1,sbl*2)];
        result[vix(t,0,sbl*2)] += sF[sfix(t,sbl,1)]*v01;
        result[vix(t,1,sbl*2)] += sF[sfix(t,sbl,6)]*v01;
        result[vix(t,2,sbl*2)] += std::conj(sF[sfix(t,sbl,7)])*v01;
        result[vix(t,0,sbl*2+1)] += std::conj(sF[sfix(t,sbl,8)])*v01;
        result[vix(t,1,sbl*2+1)] += std::conj(sF[sfix(t,sbl,9)])*v01;
        result[vix(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,10)])*v01;

        // contribution from s=0,g=2
        std::complex<double> v02 = v[vix(t,2,sbl*2)];
        result[vix(t,0,sbl*2)] += sF[sfix(t,sbl,2)]*v02;
        result[vix(t,1,sbl*2)] += sF[sfix(t,sbl,7)]*v02;
        result[vix(t,2,sbl*2)] += sF[sfix(t,sbl,11)]*v02;
        result[vix(t,0,sbl*2+1)] += std::conj(sF[sfix(t,sbl,12)])*v02;
        result[vix(t,1,sbl*2+1)] += std::conj(sF[sfix(t,sbl,13)])*v02;
        result[vix(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,14)])*v02;

        // contribution from s=1,g=0
        std::complex<double> v10 = v[vix(t,0,sbl*2+1)];
        result[vix(t,0,sbl*2)] += sF[sfix(t,sbl,3)]*v10;
        result[vix(t,1,sbl*2)] += sF[sfix(t,sbl,8)]*v10;
        result[vix(t,2,sbl*2)] += sF[sfix(t,sbl,12)]*v10;
        result[vix(t,0,sbl*2+1)] += sF[sfix(t,sbl,15)]*v10;
        result[vix(t,1,sbl*2+1)] += std::conj(sF[sfix(t,sbl,16)])*v10;
        result[vix(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,17)])*v10;

        // contribution from s=1,g=1
        std::complex<double> v11 = v[vix(t,1,sbl*2+1)];
        result[vix(t,0,sbl*2)] += sF[sfix(t,sbl,4)]*v11;
        result[vix(t,1,sbl*2)] += sF[sfix(t,sbl,9)]*v11;
        result[vix(t,2,sbl*2)] += sF[sfix(t,sbl,13)]*v11;
        result[vix(t,0,sbl*2+1)] += sF[sfix(t,sbl,16)]*v11;
        result[vix(t,1,sbl*2+1)] += sF[sfix(t,sbl,18)]*v11;
        result[vix(t,2,sbl*2+1)] += std::conj(sF[sfix(t,sbl,19)])*v11;

        // contribution from s=1,g=2
        std::complex<double> v12 = v[vix(t,2,sbl*2+1)];
        result[vix(t,0,sbl*2)] += sF[sfix(t,sbl,5)]*v12;
        result[vix(t,1,sbl*2)] += sF[sfix(t,sbl,10)]*v12;
        result[vix(t,2,sbl*2)] += sF[sfix(t,sbl,14)]*v12;
        result[vix(t,0,sbl*2+1)] += sF[sfix(t,sbl,17)]*v12;
        result[vix(t,1,sbl*2+1)] += sF[sfix(t,sbl,19)]*v12;
        result[vix(t,2,sbl*2+1)] += sF[sfix(t,sbl,20)]*v12;
    }

}

// both of the following can be inserted into the s and g loops of the wilson
// thus the function also has s and g as parameters

// clover term that takes a precomputed F_munu
// memory layout is F[munu,t,g,h] (not used in the thesis)
inline void clover_fpre_mtg (const std::complex<double> *v, std::complex<double> *result, const std::complex<double> *F,
                  int vol, double csw, int t, int g, int s){

    std::complex<double> interm (0.0,0.0);
    for (int gi = 0; gi < 3; gi++){
        for (int munu = 0; munu < 6; munu++){
            interm += F[fix(munu,t,g,gi,vol)]
                        * sigf[munu][s] * v[vix(t,gi,sigx[munu][s])];
        }
    }
    result[vix(t,g,s)] += -csw*0.5*interm;

}

// memory layout is F[t,munu,g,h]
inline void clover_fpre_tmg (const std::complex<double> *v, std::complex<double> *result, const std::complex<double> *F,
                  int vol, double csw, int t, int g, int s){

    std::complex<double> interm (0.0,0.0);
    for (int gi = 0; gi < 3; gi++){
        for (int munu = 0; munu < 6; munu++){
            interm += F[fix2(t,munu,g,gi)]
                        * sigf[munu][s] * v[vix(t,gi,sigx[munu][s])];
        }
    }
    result[vix(t,g,s)] += -csw*0.5*interm;

}



// implementations of the Wilson clover Dirac operator
// using the above clover terms and the Wilson Dirac operator with separate matrices

std::complex<double>* dwc_direct (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, double csw, int vol){

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

        clover_direct(U,v,result,hops,vol,csw,t);
    }


    return result;
}


std::complex<double>* dwc_direct_rearr (const std::complex<double>* U,
                                          const std::complex<double>* v,
                                          const int* hops, double mass, double csw, int vol){

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

        clover_direct_rearr(U,v,result,hops,vol,csw,t);
    }


    return result;
}

std::complex<double>* dwc_fpre_tmg (const std::complex<double>* U,
                                          const std::complex<double>* v, const std::complex<double> *F,
                                          const int* hops, double mass, double csw, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h], v[t,s,h] and F[t,munu,g,h]

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

                clover_fpre_tmg(v,result,F,vol,csw,t,g,s);
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

std::complex<double>* dwc_grid (const std::complex<double>* U,
                                          const std::complex<double>* v, const std::complex<double> *sF,
                                          const int* hops, double mass, int vol){

    // in this function, we use only the flattened space-time index
    // The indices for the input arrays are U[mu,t,g,h], v[t,s,h] and sigmaF[t,triangle index, block number]

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

        clover_grid(v,result,sF,vol,t);
    }


    return result;
}


// sparse Wilson clover is just a sparse matrix-vector multiplication
std::complex<double>* dwc_full (const std::complex<double>* vals,
                                const std::complex<double>* v, int * comp_ptr, int vol){

    std::complex<double>* result = new std::complex<double> [vol*4*3];

    for (int i = 0; i < vol*4*3; i++){
        std::complex<double> comp_res = 0;
        for (int j = 0; j < 54; j++){
            comp_res += vals[i*54+j]*v[comp_ptr[i*54+j]];
        }
        result[i] = comp_res;
    }

    return result;
}



int main (){
    int n_rep = 200;
    int n_warmup = 20;
    double mass = -0.5;
    double csw = 1.0;
    std::cout << n_rep << " repetitions" << std::endl;
    std::cout << "mass = " << mass << ", csw = " << csw << std::endl;
    int Lxyz[4] = {4,4,4,4};

    double max_time = 2.0e+9;
    std::cout << "computation only evaluated until time reaches " << max_time << "ns" << std::endl;
    bool do_direct = true;
    bool do_rearr = true;
    bool do_full = true;

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
        std::mt19937 generator(16);
        std::normal_distribution<double> normal(0.0, 1.0);

        int uvol = vol * 4 * n_gauge * n_gauge;
        int vvol = vol * n_gauge * n_spin;
        int fvol = vol * 6 * n_gauge * n_gauge;
        int sfvol = vol * 42;

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
        std::complex<double>* F = new std::complex<double>[fvol];
        for (int i = 0; i < fvol; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            F[i] = std::complex<double> (ree,imm);
        }
        std::complex<double>* sF = new std::complex<double>[fvol];
        for (int i = 0; i < sfvol; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            sF[i] = std::complex<double> (ree,imm);
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
        int* comp_ptr = new int [vvol*54];
        for (int t = 0; t < vol; t++){
            for (int s = 0; s < 4; s++){
                for (int g = 0; g < 3; g++){
                    // address that corresponds to t,s,g in the component lookup table
                    int currcomp = (t*12+s*3+g)*54;

                    // mass term and field strength term combined
                    // all colour components
                    // if s in {0,1}, this is all 6 components with s in {0,1}
                    // if s in {2,3}, this is all 6 with s in {2,3}
                    for (int gg = 0; gg < 6; gg++){
                        comp_ptr[currcomp+gg] = t*12+(s/2)*6+gg;
                    }
                    
                    // hop terms: iterate over mu, sign of mu, and all 3 gi
                    // there are 2 hops: with and without gamma
                    for (int mu = 0; mu < 4; mu++){
                        for (int dir = 0; dir < 2; dir++){
                            for (int gi = 0; gi < 3; gi++){
                                comp_ptr[currcomp+6+mu*12+dir*6+gi] = hops[hix(t,mu,dir)]*12+s*3+gi;
                                comp_ptr[currcomp+6+mu*12+dir*6+gi+3] = hops[hix(t,mu,dir)]*12+gamx[mu][s]*3+gi;
                            }
                        }
                    }
                }
            }
        }


        // random complex numbers as the non-zero values from the wilson clover dirac matrix
        std::complex<double>* vals = new std::complex<double>[vvol*54];
        for (int i = 0; i < vvol*54; i++){
            double ree = normal(generator);
            double imm = normal(generator);
            vals[i] = std::complex<double> (ree,imm);
        }



        std::cout << vol << " sites" << std::endl;
        std::cout << "grid layout " << Lx << " " << Ly << " " << Lz << " " << Lt << std::endl;

        double * times_direct = new double [n_rep];
        double * times_direct_r = new double [n_rep];
        double * times_fpre = new double [n_rep];
        double * times_grid = new double [n_rep];
        double * times_full = new double [n_rep];

        for (int n = 0; n < n_warmup; n++){
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[3] = randval;

            // only do warmup if max_time not reached
            if (do_direct && do_rearr && n==0){
                std::complex<double> * result = dwc_direct(U,v,hops,mass,csw,vol);
                noopt += result[3];
                std::complex<double> * result4 = dwc_direct_rearr(U,v,hops,mass,csw,vol);
                noopt += result4[6];
                int wrong = 0;
                for (int i = 0; i < vvol; i++){
                    if (abs(result[i]-result4[i]) > 0.0001){
                        wrong++;
                    }
                }
                std::cout << "wrong sites for runtime compute: " << wrong << std::endl;
                delete[] result;
                delete[] result4;
            } else {
                if (do_direct){
                    std::complex<double> * result = dwc_direct(U,v,hops,mass,csw,vol);
                    noopt += result[3];
                    delete[] result;
                }
                if (do_rearr){
                    std::complex<double> * result4 = dwc_direct_rearr(U,v,hops,mass,csw,vol);
                    noopt += result4[6];
                    delete[] result4;
                }
            }
            
            std::complex<double> * result2 = dwc_fpre_tmg(U,v,F,hops,mass,csw,vol);
            noopt += result2[13];
            delete[] result2;

            std::complex<double> * result3 = dwc_grid(U,v,sF,hops,mass,vol);
            noopt += result3[8];
            delete[] result3;

            if (do_full){
                std::complex<double> * result5 = dwc_full(vals,v,comp_ptr,vol);
                noopt += result5[45];
                delete[] result5;
            }
        }

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            double reval = normal(generator);
            double imval = normal(generator);
            std::complex<double> randval (reval,imval);
            v[33] = randval;

            // only measure until max_time is reached
            if (do_direct){
                auto start = std::chrono::high_resolution_clock::now();
                std::complex<double> * result = dwc_direct(U,v,hops,mass,csw,vol);
                auto stop = std::chrono::high_resolution_clock::now();
                auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
                times_direct[n] = double(dur.count());
                noopt += result[3];
                delete[] result;
                // if max_time is reached, in last measure iteration, set flag
                if (n == n_rep-1 && times_direct[n] > max_time){
                    do_direct = false;
                }
            } else {
                times_direct[n] = 1.0e+10;
            }
            if (do_rearr){
                auto start = std::chrono::high_resolution_clock::now();
                std::complex<double> * result4 = dwc_direct_rearr(U,v,hops,mass,csw,vol);
                auto stop = std::chrono::high_resolution_clock::now();
                auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
                times_direct_r[n] = double(dur.count());
                noopt += result4[35];
                delete[] result4;
                if (n == n_rep-1 && times_direct_r[n] > max_time){
                    do_rearr = false;
                }
            } else {
                times_direct_r[n] = 10000000000.0;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result2 = dwc_fpre_tmg(U,v,F,hops,mass,csw,vol);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_fpre[n] = double(dur.count());
            noopt += result2[13];
            delete[] result2;

            start = std::chrono::high_resolution_clock::now();
            std::complex<double> * result3 = dwc_grid(U,v,sF,hops,mass,vol);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_grid[n] = double(dur.count());
            noopt += result3[8];
            delete[] result3;

            if (do_full){
                auto start = std::chrono::high_resolution_clock::now();
                std::complex<double> * result5 = dwc_full(vals,v,comp_ptr,vol);
                auto stop = std::chrono::high_resolution_clock::now();
                auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
                times_full[n] = double(dur.count());
                noopt += result5[35];
                delete[] result5;
                if (n == n_rep-1 && times_full[n] > max_time){
                    do_full = false;
                }
            } else {
                times_full[n] = 10000000000.0;
            }

        }

        delete [] U;
        delete [] v;
        delete [] hops;
        delete [] F;
        delete [] sF;
        delete [] vals;
        delete [] comp_ptr;


        double avg_direct = 0.0;
        double avg_direct_r = 0.0;
        double avg_fpre = 0.0;
        double avg_grid = 0.0;
        double avg_full = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_direct,times_direct+n_rep);
        std::sort(times_direct_r,times_direct_r+n_rep);
        std::sort(times_fpre,times_fpre+n_rep);
        std::sort(times_grid,times_grid+n_rep);
        std::sort(times_full,times_full+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_direct += times_direct[n];
            avg_direct_r += times_direct_r[n];
            avg_fpre += times_fpre[n];
            avg_grid += times_grid[n];
            avg_full+= times_full[n];
        }

        avg_direct /= double(n_res);
        avg_direct_r /= double(n_res);
        avg_fpre /= double(n_res);
        avg_grid /= double(n_res);
        avg_full /= double(n_res);

        std::cout << "direct explicit computation time: " << avg_direct/1000 << " µs\n";
        std::cout << "rearranged direct explicit computation time: " << avg_direct_r/1000 << " µs\n";
        std::cout << "field strength precomputed time: " << avg_fpre/1000 << " µs\n";
        std::cout << "sigma field strength precomputed time: " << avg_grid/1000 << " µs\n";
        std::cout << "full_sparse matrix time: " << avg_full/1000 << " µs\n";

        delete [] times_direct;
        delete [] times_direct_r;
        delete [] times_fpre;
        delete [] times_grid;
        delete [] times_full;
    }

    std::cout << "print noopt to prevent optimization: " << noopt << std::endl;
}
