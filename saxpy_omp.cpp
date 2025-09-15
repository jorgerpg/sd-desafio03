// saxpy_omp.cpp
// Compile: g++ -O3 -std=c++17 -fopenmp saxpy_omp.cpp -o saxpy_omp
// Run:     OMP_NUM_THREADS=8 ./saxpy_omp [N]
// Ex.:     OMP_NUM_THREADS=8 ./saxpy_omp 50000000

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>

int main(int argc, char** argv) {
    using clock = std::chrono::high_resolution_clock;

    size_t N = (argc > 1) ? std::stoull(argv[1]) : 50'000'000ULL;
    const float a = 2.0f;

    int nthreads = omp_get_max_threads();
    std::cout << "CPU OpenMP | N=" << N << " | threads=" << nthreads << "\n";

    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float* y_ref = (float*)malloc(N * sizeof(float));
    if (!x || !y || !y_ref) { std::cerr << "Falha ao alocar memoria.\n"; return 1; }

    // Inicialização determinística
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        x[i] = (float)((i % 1000) - 500) / 500.0f;
        y[i] = (float)((i % 777) - 388) / 388.0f;
        y_ref[i] = y[i];
    }

    auto t0 = clock::now();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        y_ref[i] = a * x[i] + y_ref[i];
    }

    auto t1 = clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Métricas
    const double flops = 2.0 * (double)N;
    double gflops = (flops / 1e9) / (ms / 1e3);

    // Cálculo do checksum de forma consistente
    double checksum = 0.0;
    for (int k = 0; k < 10 && (size_t)k < N; ++k) {
        checksum += y_ref[k];
    }

    std::cout << "Tempo: " << ms << " ms | " << gflops << " GFLOP/s\n";
    std::cout << "Checksum(10): " << checksum << "\n";

    free(x); free(y); free(y_ref);
    return 0;
}