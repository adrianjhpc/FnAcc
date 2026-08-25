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

__global__ void daxpy_kernel(double alpha, const double *x, double *y,
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

  double alpha = 3.0;

  std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);

  double *x = nullptr;
  double *y = nullptr;
  double *y0 = nullptr;

  CHECK(cudaMallocHost(&x, bytes));
  CHECK(cudaMallocHost(&y, bytes));
  CHECK(cudaMallocHost(&y0, bytes));

  for (long long i = 0; i < n; ++i) {
    int v = static_cast<int>((i + 1) % 1000);
    x[i] = 0.001 * static_cast<double>(v);
    y[i] = 10.0 + 0.002 * static_cast<double>(v);
    y0[i] = y[i];
  }

  double *dx = nullptr;
  double *dy = nullptr;

  CHECK(cudaMalloc(&dx, bytes));
  CHECK(cudaMalloc(&dy, bytes));

  CHECK(cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy, y, bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);

  daxpy_kernel<<<grid, block>>>(alpha, dx, dy, n);
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
    daxpy_kernel<<<grid, block>>>(alpha, dx, dy, n);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  CHECK(cudaGetLastError());

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaMemcpy(y, dy, bytes, cudaMemcpyDeviceToHost));

  int errors = 0;
  for (long long i = 0; i < n; ++i) {
    double expected = y0[i] + static_cast<double>(reps) * alpha * x[i];
    double scale = fmax(1.0, fmax(fabs(y[i]), fabs(expected)));
    double tol = fmax(1.0e-10, 1.0e-12 * scale);

    if (fabs(y[i] - expected) > tol) {
      if (errors < 10) {
        std::fprintf(stderr, "mismatch at %lld: got %.17g expected %.17g\n",
                     i + 1, y[i], expected);
      }
      ++errors;
    }
  }

  double avgSeconds = (static_cast<double>(ms) / 1.0e3) / reps;
  double bytesPerRep = 3.0 * static_cast<double>(n) * sizeof(double);
  double gbps = bytesPerRep / avgSeconds / 1.0e9;

  std::printf("cuda_daxpy,%lld,1,%d,%.8e,%.8e,%d\n", n, reps, avgSeconds,
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

