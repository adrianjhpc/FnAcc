#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t e, const char *expr, const char *file, int line) {
  if (e == cudaSuccess)
    return;

  std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file,
               line, expr, cudaGetErrorString(e));
  std::abort();
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

__global__ void saxpy_kernel(float alpha, const float *x, float *y,
                             long long n) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    y[i] = alpha * x[i] + y[i];
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

int main(int argc, char **argv) {
  long long n = parse_i64(argc, argv, 1, 1048576);
  int reps = parse_i32(argc, argv, 2, 100);

  float alpha = 3.0f;

  std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  float *x = nullptr;
  float *y = nullptr;
  float *y0 = nullptr;

  CHECK(cudaMallocHost(&x, bytes));
  CHECK(cudaMallocHost(&y, bytes));
  CHECK(cudaMallocHost(&y0, bytes));

  for (long long i = 0; i < n; ++i) {
    int v = static_cast<int>((i + 1) % 1000);
    x[i] = 0.001f * static_cast<float>(v);
    y[i] = 10.0f + 0.002f * static_cast<float>(v);
    y0[i] = y[i];
  }

  float *dx = nullptr;
  float *dy = nullptr;

  CHECK(cudaMalloc(&dx, bytes));
  CHECK(cudaMalloc(&dy, bytes));

  CHECK(cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy, y, bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);

  saxpy_kernel<<<grid, block>>>(alpha, dx, dy, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Reset y before timed loop so correctness check has a clean reference.
  CHECK(cudaMemcpy(dy, y0, bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start;
  cudaEvent_t stop;

  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r)
    saxpy_kernel<<<grid, block>>>(alpha, dx, dy, n);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaMemcpy(y, dy, bytes, cudaMemcpyDeviceToHost));

  int errors = 0;
  for (long long i = 0; i < n; ++i) {
    float expected =
      y0[i] + static_cast<float>(reps) * alpha * x[i];

    float scale = fmaxf(1.0f, fmaxf(fabsf(y[i]), fabsf(expected)));
    float tol = fmaxf(1.0e-3f, 1.0e-4f * scale);
 
    if (std::fabs(y[i] - expected) > tol) {
      if (errors < 10) {
        std::fprintf(stderr, "mismatch at %lld: got %f expected %f\n", i + 1,
                     y[i], expected);
      }
      ++errors;
    }
  }

  double avgSeconds = (static_cast<double>(ms) / 1.0e3) / reps;
  double bytesPerRep = 3.0 * static_cast<double>(n) * sizeof(float);
  double gbps = bytesPerRep / avgSeconds / 1.0e9;

  std::printf("cuda_saxpy,%lld,1,%d,%.8e,%.8e,%d\n", n, reps, avgSeconds,
              gbps, errors);

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  CHECK(cudaFree(dx));
  CHECK(cudaFree(dy));

  CHECK(cudaFreeHost(x));
  CHECK(cudaFreeHost(y));
  CHECK(cudaFreeHost(y0));

  return errors == 0 ? 0 : 1;
}

