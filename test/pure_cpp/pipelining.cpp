// simple code example to showcase pipelining

#include <complex>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>


// sum over single array
void array_sum (double * a, int len, double * c){
    double res = 0.0;
#pragma GCC novector
    for (int i = 0; i < len; i++){
        res += a[i];
    }
    c[0] = res;
}

// sum over 2 arrays at the same time
void two_array_sum (double * a, double * b, int len, double * c){
    double resa = 0.0;
    double resb = 0.0;
#pragma GCC novector
    for (int i = 0; i < len; i++){
        resa += a[i];
        resb += b[i];
    }
    c[0] = resa;
    c[1] = resb;
}


int main (){
    int n_rep = 200;
    std::cout << "sum over double numbers, " << n_rep << " repetitions, best 20%" << std::endl;

    int len = 16*16*8*2;

    // add results to this to prevent optimization
    double cx1 = 0.0;
    double cx2 = 0.0;
    double cx3 = 0.0;

    for (int len_vers = 0; len_vers < 12; len_vers++){
        len *= 2;
        std::cout << "length " << len << std::endl;

        std::mt19937 generator(3);
        std::normal_distribution<double> normal(0.0, 1.0);

        double * a = new double [len];
        double * b = new double [len];
        double * d = new double [len];

        double c[2];

        for (int i = 0; i < len; i++){
            a[i] = normal(generator);
            b[i] = normal(generator);
            d[i] = normal(generator);
        }

        double * times_1 = new double [n_rep];
        double * times_2 = new double [n_rep];

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            a[1] = normal(generator);
            b[33] = normal(generator);
            d[4] = normal(generator);

            auto start = std::chrono::high_resolution_clock::now();
            array_sum(a, len, c);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_1[n] = double(dur.count());
            cx1 += c[0];
            
            start = std::chrono::high_resolution_clock::now();
            two_array_sum(b, d, len, c);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_2[n] = double(dur.count());
            cx2 += c[0];
            cx3 += c[1];
        }

        delete [] a;
        delete [] b;
        delete [] d;

        double avg_1 = 0.0;
        double avg_2 = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_1,times_1+n_rep);
        std::sort(times_2,times_2+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_1 += times_1[n];
            avg_2 += times_2[n];
        }

        avg_1 /= double(n_res);
        avg_2 /= double(n_res);

        std::cout << "1 array time: " << avg_1/1000 << " µs\n";
        std::cout << "2 array time: " << avg_2/1000 << " µs\n";
        
        delete [] times_1;
        delete [] times_2;
    }
    std::cout << "print sums to prevent optimization: " << cx1 << ", " << cx2 << ", " << cx3 << std::endl;
}

