// simple code example to showcase parallelization

#include <omp.h>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>



double array_sum (double * a, int len){
    double res = 0.0;
    for (int i = 0; i < len; i++){
        res += a[i];
    }
    return res;
}

double parallel_array_sum (double * a, int len){
    double res = 0.0;
#pragma omp parallel for reduction (+:res)
    for (int i = 0; i < len; i++){
        res += a[i];
    }
    return res;
}

void array_add (double *a, double *b, int len){
    for (int i = 0; i < len; i++){
        a[i] = a[i] + b[i];
    }
}

void parallel_array_add (double *a, double *b, int len){
#pragma omp parallel for
    for (int i = 0; i < len; i++){
        a[i] = a[i] + b[i];
    }
}


int main (){
    int n_rep = 200;
    std::cout << "sum over double numbers, " << n_rep << " repetitions, best 20%" << std::endl;

    int len = 16*16*8*2;

    // add results to this to prevent optimization
    double cx1 = 0.0;
    double cx2 = 0.0;

    for (int len_vers = 0; len_vers < 12; len_vers++){
        len *= 2;
        std::cout << "length " << len << std::endl;

        std::mt19937 generator(51);
        std::normal_distribution<double> normal(0.0, 0.2);

        double * a = new double [len];
        double * b = new double [len];

        for (int i = 0; i < len; i++){
            a[i] = normal(generator);
            b[i] = normal(generator);
        }

        double * times_a = new double [n_rep];
        double * times_ap = new double [n_rep];
        double * times_s = new double [n_rep];
        double * times_sp = new double [n_rep];

        for (int n = 0; n < n_rep; n++){
            // small change in each repetition
            a[14] = normal(generator);
            double c1, c2;

            auto start = std::chrono::high_resolution_clock::now();
            array_add(a, b, len);
            auto stop = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_a[n] = double(dur.count());

            start = std::chrono::high_resolution_clock::now();
            parallel_array_add(a, b, len);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_ap[n] = double(dur.count());

            start = std::chrono::high_resolution_clock::now();
            c1 = array_sum(a, len);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_s[n] = double(dur.count());
            cx1 += c1;
                        
            start = std::chrono::high_resolution_clock::now();
            c2 = parallel_array_sum(a, len);
            stop = std::chrono::high_resolution_clock::now();
            dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
            times_sp[n] = double(dur.count());
            cx2 += c2;

            if (n==0 && abs(c1-c2)>0.001){
                std::cout << "sum wrong" << std::endl;
            }
            
        }

        delete [] a;
        delete [] b;

        double avg_a = 0.0;
        double avg_ap = 0.0;
        double avg_s = 0.0;
        double avg_sp = 0.0;

        // only consider best 20% of results
        int n_res = n_rep/5;

        std::sort(times_a,times_a+n_rep);
        std::sort(times_ap,times_ap+n_rep);
        std::sort(times_s,times_s+n_rep);
        std::sort(times_sp,times_sp+n_rep);


        for (int n = 0; n < n_res; n++){
            avg_a += times_a[n];
            avg_ap += times_ap[n];
            avg_s += times_s[n];
            avg_sp += times_sp[n];
        }

        avg_a /= double(n_res);
        avg_ap /= double(n_res);
        avg_s /= double(n_res);
        avg_sp /= double(n_res);

        std::cout << "add_normal time: " << avg_a/1000 << " µs\n";
        std::cout << "add_parallel time: " << avg_ap/1000 << " µs\n";
        std::cout << "sum_normal time: " << avg_s/1000 << " µs\n";
        std::cout << "sum_parallel time: " << avg_sp/1000 << " µs\n";
        
        delete [] times_a;
        delete [] times_ap;
        delete [] times_s;
        delete [] times_sp;
    }
    std::cout << "print sums to prevent optimization: " << cx1 << ", " << cx2 << std::endl;
}

