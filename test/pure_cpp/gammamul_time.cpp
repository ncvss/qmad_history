// test the gamma matrix index lookup against standard multiplication

#include <complex>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>



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
    

static const std::complex<double> gamma_y [4][4] = {{0,0,0,-1},{0,0,1,0},{0,1,0,0},{-1,0,0,0}};


// iterate over the arrays vec and res, treat 4 array entries each as a spin vector
// apply the gamma y matrix alternatingly to res and store the result in the other array
// application to multiple vectors reduces influence of latency
// repeated application to the same data so that bandwidth does not become the bottleneck
void standard_matmul (std::complex<double>* vec, std::complex<double>* res, int its, int inner){
    for (int p = 0; p < its*4; p+=4){
        for (int app = 0; app < inner; app++){
            for (int i = 0; i < 4; i++){
                std::complex<double> sum_up = 0;
                for (int j = 0; j < 4; j++){
                    sum_up += gamma_y[i][j] * vec[p+j];
                }
                res[p+i] = sum_up;
            }
            for (int i = 0; i < 4; i++){
                std::complex<double> sum_up = 0;
                for (int j = 0; j < 4; j++){
                    sum_up += gamma_y[i][j] * res[p+j];
                }
                vec[p+i] = sum_up;
            }
        }
    }
}


void lookup_matmul (std::complex<double>* vec, std::complex<double>* res, int its, int inner){
    for (int p = 0; p < its*4; p+=4){
        for (int app = 0; app < inner; app++){
            for (int i = 0; i < 4; i++){
                res[p+i] = gamf[1][i]*vec[p+gamx[1][i]];
            }
            for (int i = 0; i < 4; i++){
                vec[p+i] = gamf[1][i]*res[p+gamx[1][i]];
            }
        }
    }
}


int main (){

    // number of vectors that are multiplied
    const int n_its = 16*16*8;
    // number of times the matrix is multiplied onto the vector
    const int n_inner_its = 256;
    const int n_repetitions = 500;
    std::cout << n_its << " vectors, " << n_inner_its << " applications of gamma_y, " << n_repetitions << " repetitions of measurement\n";
    // only top 20% of results for evaluation
    const int n_results = n_repetitions/5;
    const int n_data = n_its*4;

    std::complex<double> * vec1 = new std::complex<double> [n_data];
    std::complex<double> * vec2 = new std::complex<double> [n_data];
    std::complex<double> * res1 = new std::complex<double> [n_data];
    std::complex<double> * res2 = new std::complex<double> [n_data];

    // random numbers
    std::mt19937 generator(73);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (int i = 0; i < n_data; i++){
        double x = normal(generator);
        double y = normal(generator);
        vec1[i] = std::complex<double>(x,y);
        vec2[i] = std::complex<double>(x,y);
    }

    double * time_stand = new double [n_repetitions];
    double * time_look = new double [n_repetitions];


    for (int j = 0; j < n_repetitions; j++){

        // small change
        double ch = normal(generator);
        vec1[0] = std::complex<double>(0.1,ch);
        vec2[0] = std::complex<double>(0.1,ch);

        auto start = std::chrono::high_resolution_clock::now();
        standard_matmul(vec1, res1, n_its, n_inner_its);
        auto stop = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
        time_stand[j] = double(dur.count());

        start = std::chrono::high_resolution_clock::now();
        lookup_matmul(vec2, res2, n_its, n_inner_its);
        stop = std::chrono::high_resolution_clock::now();
        dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
        time_look[j] = double(dur.count());

    }

    // check
    int equality = 0;
    for (int i = 0; i < n_data; i++){
        if (std::abs(std::real(res1[i])-std::real(res2[i]))>0.0001 || std::abs(std::imag(res1[i])-std::imag(res2[i]))>0.0001){
            equality++;
        }
    }
    if (equality == 0){
        std::cout << "computations correct" << std::endl;
    } else {
        std::cout << "computations incorrect at " << equality << " points" << std::endl;
    }

    delete [] vec1;
    delete [] vec2;
    delete [] res1;
    delete [] res2;

    double avg_stand = 0;
    double avg_look = 0;

    std::sort(time_stand, time_stand+n_repetitions);
    std::sort(time_look, time_look+n_repetitions);

    for (int j = 0; j < n_results; j++){
        avg_stand += time_stand[j];
        avg_look += time_look[j];
    }

    avg_stand /= double(n_results);
    avg_look /= double(n_results);

    std::cout << "standard matmul time in ns: " << avg_stand << "\n";
    std::cout << "index lookup time in ns: " << avg_look << std::endl;

    delete [] time_stand;
    delete [] time_look;

}
