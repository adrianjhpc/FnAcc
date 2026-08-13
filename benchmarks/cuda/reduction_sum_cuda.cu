#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t e, const char *expr, const char *file, int line) {
  if (e == cudaSuccess)
    return;

  std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file,
               line, expr, cudaGetErrorString(e));
  std::abort();
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

static double wall_time() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static long long parse_i64(int argc, char **argv, int index,
                           long long fallback) {
  if (argc <= index)
    return fallback;
  return std::atoll(argv[index]);
}

static int parse_i32(int argc, char **argv, int index, int fallback) {
  if (argc <= index)
    return fallback;
  return std::atoi(argv[index]);
}

__global__ void reduction_sum_kernel(const float *a, float *partials,
                                     long long n) {
  extern __shared__ float sh[];

  int tid = threadIdx.x;
  long long i = blockIdx.x * blockDim.x + threadIdx.x;

  float v = 0.0f;
  if (i < n)
    v = a[i];

  sh[tid] = v;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      sh[tid] += sh[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    partials[blockIdx.x] = sh[0];
}

int main(int argc, char **argv) {
  long long n = parse_i64(argc, argv, 1, 1048576);
  int reps = parse_i32(argc, argv, 2, 100);

  std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  float *a = nullptr;
  CHECK(cudaMallocHost(&a, bytes));

  for (long long i = 0; i < n; ++i)
    a[i] = 0.001f * static_cast<float>(i % 1000);

  float *da = nullptr;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);

  float *dpartials = nullptr;
  std::size_t partialBytes = static_cast<std::size_t>(grid) * sizeof(float);
  CHECK(cudaMalloc(&dpartials, partialBytes));

  std::vector<float> partials(grid);

  // Full-operation warm-up.
  reduction_sum_kernel<<<grid, block, block * sizeof(float)>>>(da, dpartials,
                                                               n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(partials.data(), dpartials, partialBytes,
                   cudaMemcpyDeviceToHost));

  float lastResult = 0.0f;

  double t0 = wall_time();

  for (int r = 0; r < reps; ++r) {
    reduction_sum_kernel<<<grid, block, block * sizeof(float)>>>(da, dpartials,
                                                                 n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(partials.data(), dpartials, partialBytes,
                     cudaMemcpyDeviceToHost));

    float result = 0.0f;
    for (float v : partials)
      result += v;

    lastResult = result;
  }

  double t1 = wall_time();

  double expected = 0.0;
  for (long long i = 0; i < n; ++i)
    expected += static_cast<double>(a[i]);

  double diff = fabs(static_cast<double>(lastResult) - expected);
  double tol = 1.0e-2 + 1.0e-4 * fabs(expected);

  int errors = 0;
  if (diff > tol) {
    std::fprintf(stderr,
                 "sum mismatch got %.8e expected %.8e diff %.8e tol %.8e\n",
                 static_cast<double>(lastResult), expected, diff, tol);
    errors = 1;
  }

  double avgSeconds = (t1 - t0) / reps;
  double bytesPerRep = static_cast<double>(n) * sizeof(float);
  double gbps = bytesPerRep / avgSeconds / 1.0e9;

  std::printf("cuda_reduction_sum,%lld,1,%d,%.8e,%.8e,%d\n", n, reps,
              avgSeconds, gbps, errors);

  CHECK(cudaFree(dpartials));
  CHECK(cudaFree(da));
  CHECK(cudaFreeHost(a));

  return errors == 0 ? 0 : 1;
}

