// saxpy_cuda.cu
// Compile: nvcc -O3 -std=c++17 saxpy_cuda.cu -o saxpy_cuda
// Run:     ./saxpy_cuda [N] [use_pinned]
// Ex.:     ./saxpy_cuda 50000000 1    // 50M elementos, host pinned (mais rápido nas cópias)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

__global__ void saxpy_kernel(const float a, const float* __restrict__ x, float* __restrict__ y, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

int main(int argc, char** argv) {
  size_t N = (argc > 1) ? std::stoull(argv[1]) : 50'000'000ULL;
  bool use_pinned = (argc > 2) ? (std::atoi(argv[2]) != 0) : false;
  const float a = 2.0f;

  std::cout << "CUDA | N=" << N << " | host=" << (use_pinned ? "pinned" : "pageable") << "\n";

  float *h_x = nullptr, *h_y = nullptr, *h_y_out = nullptr;
  if (use_pinned) {
    CUDA_CHECK(cudaMallocHost(&h_x,     N*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_y,     N*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_y_out, N*sizeof(float)));
  } else {
    h_x     = (float*)std::malloc(N*sizeof(float));
    h_y     = (float*)std::malloc(N*sizeof(float));
    h_y_out = (float*)std::malloc(N*sizeof(float));
  }
  if (!h_x || !h_y || !h_y_out) { std::cerr << "Falha ao alocar host.\n"; return 1; }

  // Inicialização determinística
  for (size_t i = 0; i < N; ++i) {
    h_x[i] = (float)((i % 1000) - 500) / 500.0f;
    h_y[i] = (float)((i % 777) - 388) / 388.0f;
  }

  float *d_x=nullptr, *d_y=nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(float)));

  cudaEvent_t ev_start, ev_after_h2d, ev_after_kernel, ev_end;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_after_h2d));
  CUDA_CHECK(cudaEventCreate(&ev_after_kernel));
  CUDA_CHECK(cudaEventCreate(&ev_end));

  // Warm-up
  CUDA_CHECK(cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice));
  
  const int block = 256;
  const int grid  = (int)std::min((size_t)131072, (N + block - 1) / block);
  saxpy_kernel<<<grid, block>>>(a, d_x, d_y, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  CUDA_CHECK(cudaEventRecord(ev_start));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(ev_after_h2d));

  saxpy_kernel<<<grid, block>>>(a, d_x, d_y, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev_after_kernel));

  CUDA_CHECK(cudaMemcpy(h_y_out, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(ev_end));
  CUDA_CHECK(cudaEventSynchronize(ev_end));

  float ms_total=0, ms_h2d=0, ms_kernel=0, ms_d2h=0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev_start, ev_end));
  CUDA_CHECK(cudaEventElapsedTime(&ms_h2d,   ev_start, ev_after_h2d));
  CUDA_CHECK(cudaEventElapsedTime(&ms_kernel,ev_after_h2d, ev_after_kernel));
  CUDA_CHECK(cudaEventElapsedTime(&ms_d2h,   ev_after_kernel, ev_end));

  // Verificação rápida de erro contra referência CPU simples (amostra)
  double max_abs_err = 0.0;
  for (int k = 0; k < 1000 && (size_t)k < N; ++k) {
    float ref = a * h_x[k] + h_y[k];
    max_abs_err = std::max(max_abs_err, (double)std::abs(ref - h_y_out[k]));
  }

  // Métricas
  const double flops = 2.0 * (double)N;     // 1 mult + 1 soma
  const double bytes = 12.0 * (double)N;    // ler x (4), ler y (4), escrever y (4)

  double gflops_kernel = (flops / 1e9) / (ms_kernel / 1e3);
  double gbps_total    = (bytes / 1e9) / (ms_total / 1e3);

  std::cout << "H2D: " << ms_h2d << " ms | K: " << ms_kernel << " ms | D2H: " << ms_d2h 
            << " ms | Total: " << ms_total << " ms\n";
  std::cout << "Kernel: " << gflops_kernel << " GFLOP/s | Total BW ~ " << gbps_total << " GB/s\n";
  std::cout << "Max |err| (amostra 1k): " << max_abs_err << "\n";

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_after_h2d));
  CUDA_CHECK(cudaEventDestroy(ev_after_kernel));
  CUDA_CHECK(cudaEventDestroy(ev_end));

  if (use_pinned) {
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));
    CUDA_CHECK(cudaFreeHost(h_y_out));
  } else {
    std::free(h_x);
    std::free(h_y);
    std::free(h_y_out);
  }
  return 0;
}