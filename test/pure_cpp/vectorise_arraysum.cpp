// code example to showcase vectorisation

#include <complex>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <immintrin.h>


// sum over single array
void array_sum (double * a, int len, double * c){
    double res = 0.0;
    for (int i = 0; i < len; i++){
        res += a[i];
    }
    c[0] = res;
}

void unroll_array_sum (double * a, int len, double * c){
    double res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    for (int i = 0; i < len; i+=4){
        res1 += a[i];
        res2 += a[i+1];
        res3 += a[i+2];
        res4 += a[i+3];
    }
    c[0] = res1+res2+res3+res4;
}

// sum array by vectorising
void vector_array_sum (double * a, int len, double * c){
    __m256d sum4 = _mm256_setzero_pd();
    for (int i = 0; i < len; i+=4){
        __m256d a4 = _mm256_loadu_pd(a+i);
        sum4 = _mm256_add_pd(sum4, a4);
    }
    double res4 [4];
    _mm256_storeu_pd(res4,sum4);
    c[0] = res4[0]+res4[1]+res4[2]+res4[3];
}


int main (){
    int n_rep = 200;
    std::cout << "sum over double numbers, " << n_rep << " repetitions, best 20%" << std::endl;

    int len = 16*16*8*2;

    // add results to this to prevent optimization
    double cx1 = 0.0;
    double cx2 = 0.0;
    double cxu = 0.0;

    for (int len_vers = 0; len_vers < 12; len_vers++){
        len *= 2;
        std::cout << "length " << len << std::endl;

        std::mt19937 generator(53);
        std::normal_distribution<double> normal(0.0, 1.0);

        double * a = new double [len];

        double c[1];

        for (int i = 0; i < len; i++){
            a[i] = normal(generator);
        }

        double * times_1 = new double [n_rep];
        double * times_2 = new double [n_rep];
        double * times_u = new double [n_rep];

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            a[14] = normal(generator);

            auto start = std::chrono::high_resolution_clock::now();
            array_sum(a, len, c);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_1[n] = double(dur.count());
            cx1 += c[0];
            
            start = std::chrono::high_resolution_clock::now();
            vector_array_sum(a, len, c);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_2[n] = double(dur.count());
            cx2 += c[0];

            start = std::chrono::high_resolution_clock::now();
            unroll_array_sum(a, len, c);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_u[n] = double(dur.count());
            cxu += c[0];
        }

        delete [] a;

        double avg_1 = 0.0;
        double avg_2 = 0.0;
        double avg_u = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_1,times_1+n_rep);
        std::sort(times_2,times_2+n_rep);
        std::sort(times_u,times_u+n_rep);

        for (int n = 0; n < n_res; n++){
            avg_1 += times_1[n];
            avg_2 += times_2[n];
            avg_u += times_u[n];
        }

        avg_1 /= double(n_res);
        avg_2 /= double(n_res);
        avg_u /= double(n_res);

        std::cout << "normal time: " << avg_1/1000 << " µs\n";
        std::cout << "vectorised time: " << avg_2/1000 << " µs\n";
        std::cout << "unrolled time: " << avg_u/1000 << " µs\n";
        
        delete [] times_1;
        delete [] times_2;
        delete [] times_u;
    }
    std::cout << "print sums to prevent optimization: " << cx1 << ", " << cx2 << ", " << cxu << std::endl;
}

