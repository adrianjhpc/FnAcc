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

__global__ void matrix_add_2d_f64_kernel(const double *a, const double *b,
                                         double *c, long long n, long long m) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  long long j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= n || j >= m)
    return;

  // Fortran column-major layout: offset = i + j*n.
  long long offset = i + j * n;
  c[offset] = a[offset] + b[offset];
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
  long long n = parse_i64(argc, argv, 1, 1024);
  long long m = parse_i64(argc, argv, 2, 1024);
  int reps = parse_i32(argc, argv, 3, 100);

  std::size_t elements = static_cast<std::size_t>(n * m);
  std::size_t bytes = elements * sizeof(double);

  double *a = nullptr;
  double *b = nullptr;
  double *c = nullptr;

  CHECK(cudaMallocHost(&a, bytes));
  CHECK(cudaMallocHost(&b, bytes));
  CHECK(cudaMallocHost(&c, bytes));

  for (long long j = 0; j < m; ++j) {
    for (long long i = 0; i < n; ++i) {
      long long idx = i + j * n;
      a[idx] = static_cast<double>(i + 1) + 10.0 * static_cast<double>(j + 1);
      b[idx] = 2.0 * static_cast<double>(i + 1) - static_cast<double>(j + 1);
      c[idx] = 0.0;
    }
  }

  double *da = nullptr;
  double *db = nullptr;
  double *dc = nullptr;

  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));

  CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16, 1);
  dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
            static_cast<unsigned>((m + block.y - 1) / block.y), 1);

  matrix_add_2d_f64_kernel<<<grid, block>>>(da, db, dc, n, m);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;

  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r)
    matrix_add_2d_f64_kernel<<<grid, block>>>(da, db, dc, n, m);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  CHECK(cudaGetLastError());

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));

  int errors = 0;
  for (long long j = 0; j < m; ++j) {
    for (long long i = 0; i < n; ++i) {
      long long idx = i + j * n;
      double expected = a[idx] + b[idx];
      double tol = fmax(1.0e-10, 1.0e-12 * fabs(expected));

      if (fabs(c[idx] - expected) > tol) {
        if (errors < 10) {
          std::fprintf(stderr,
                       "mismatch at (%lld,%lld): got %.17g expected %.17g\n",
                       i + 1, j + 1, c[idx], expected);
        }
        ++errors;
      }
    }
  }

  double avgSeconds = (static_cast<double>(ms) / 1.0e3) / reps;
  double bytesPerRep =
      3.0 * static_cast<double>(n) * static_cast<double>(m) * sizeof(double);
  double gbps = bytesPerRep / avgSeconds / 1.0e9;

  std::printf("cuda_matrix_add_2d_f64,%lld,%lld,%d,%.8e,%.8e,%d\n", n, m,
              reps, avgSeconds, gbps, errors);

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  CHECK(cudaFree(da));
  CHECK(cudaFree(db));
  CHECK(cudaFree(dc));

  CHECK(cudaFreeHost(a));
  CHECK(cudaFreeHost(b));
  CHECK(cudaFreeHost(c));

  return errors == 0 ? 0 : 1;
}

