// saxpy_seq.cpp
// Compile: g++ -O3 -std=c++17 saxpy_seq.cpp -o saxpy_seq
// Run:     ./saxpy_seq [N]
// Ex.:     ./saxpy_seq 50000000

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>

int main(int argc, char **argv)
{
    using clock = std::chrono::high_resolution_clock;

    size_t N = (argc > 1) ? std::stoull(argv[1]) : 50'000'000ULL;
    const float a = 2.0f;

    std::cout << "CPU seq | N=" << N << "\n";

    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    float *y_ref = (float *)malloc(N * sizeof(float));
    if (!x || !y || !y_ref)
    {
        std::cerr << "Falha ao alocar memoria.\n";
        return 1;
    }

    // Inicialização determinística (barata, sem overhead de rand/sin)
    for (size_t i = 0; i < N; ++i)
    {
        x[i] = (float)((i % 1000) - 500) / 500.0f; // [-1, 1]
        y[i] = (float)((i % 777) - 388) / 388.0f;  // ~[-1, 1]
        y_ref[i] = y[i];
    }

    auto t0 = clock::now();
    for (size_t i = 0; i < N; ++i)
    {
        y_ref[i] = a * x[i] + y_ref[i];
    }
    auto t1 = clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Métricas
    const double flops = 2.0 * (double)N; // 1 mult + 1 soma
    double gflops = (flops / 1e9) / (ms / 1e3);

    // Pequena amostra do resultado e checksum
    double checksum = 0.0;
    for (int k = 0; k < 10 && (size_t)k < N; ++k)
    {
        checksum += y_ref[k];
    }

    std::cout << "Tempo: " << ms << " ms | " << gflops << " GFLOP/s\n";
    std::cout << "Checksum(10): " << checksum << "\n";

    free(x);
    free(y);
    free(y_ref);
    return 0;
}