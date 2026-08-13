#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check_cuda(cudaError_t e, const char *expr, const char *file,
                       int line) {
  if (e == cudaSuccess)
    return;

  std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file,
               line, expr, cudaGetErrorString(e));
  std::abort();
}

static const char *cublas_status_string(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "UNKNOWN_CUBLAS_STATUS";
  }
}

static void check_cublas(cublasStatus_t e, const char *expr, const char *file,
                         int line) {
  if (e == CUBLAS_STATUS_SUCCESS)
    return;

  std::fprintf(stderr, "cuBLAS error at %s:%d while executing %s: %s\n", file,
               line, expr, cublas_status_string(e));
  std::abort();
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CHECK_CUBLAS(expr) check_cublas((expr), #expr, __FILE__, __LINE__)

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
  long long n_ll = parse_i64(argc, argv, 1, 512);
  long long m_ll = parse_i64(argc, argv, 2, 512);
  int reps = parse_i32(argc, argv, 3, 20);

  // Script-compatible choice:
  //
  //   executable n m reps
  //
  // Use k = n by default.
  long long k_ll = n_ll;

  if (n_ll <= 0 || m_ll <= 0 || k_ll <= 0) {
    std::fprintf(stderr, "n, m, and k must be positive\n");
    return 1;
  }

  if (n_ll > INT_MAX || m_ll > INT_MAX || k_ll > INT_MAX) {
    std::fprintf(stderr,
                 "cuBLAS GEMM dimensions must fit in int for this benchmark\n");
    return 1;
  }

  int n = static_cast<int>(n_ll);
  int m = static_cast<int>(m_ll);
  int k = static_cast<int>(k_ll);

  std::size_t elementsA = static_cast<std::size_t>(n_ll * k_ll);
  std::size_t elementsB = static_cast<std::size_t>(k_ll * m_ll);
  std::size_t elementsC = static_cast<std::size_t>(n_ll * m_ll);

  std::size_t bytesA = elementsA * sizeof(double);
  std::size_t bytesB = elementsB * sizeof(double);
  std::size_t bytesC = elementsC * sizeof(double);

  double *a = nullptr;
  double *b = nullptr;
  double *c = nullptr;

  CHECK_CUDA(cudaMallocHost(&a, bytesA));
  CHECK_CUDA(cudaMallocHost(&b, bytesB));
  CHECK_CUDA(cudaMallocHost(&c, bytesC));

  // Column-major initialisation matching the Fortran benchmarks:
  //
  //   A(i,p) offset = i + p*n      A shape: n x k
  //   B(p,j) offset = p + j*k      B shape: k x m
  //   C(i,j) offset = i + j*n      C shape: n x m
  //
  // Here i, j, p are zero-based C indices.
  for (long long p = 0; p < k_ll; ++p) {
    for (long long i = 0; i < n_ll; ++i) {
      long long idx = i + p * n_ll;
      a[idx] = 0.001 * static_cast<double>((i + p) % 100);
    }
  }

  for (long long j = 0; j < m_ll; ++j) {
    for (long long p = 0; p < k_ll; ++p) {
      long long idx = p + j * k_ll;
      b[idx] = 0.002 * static_cast<double>((p + 2 * j) % 100);
    }
  }

  for (std::size_t idx = 0; idx < elementsC; ++idx)
    c[idx] = 0.0;

  double *da = nullptr;
  double *db = nullptr;
  double *dc = nullptr;

  CHECK_CUDA(cudaMalloc(&da, bytesA));
  CHECK_CUDA(cudaMalloc(&db, bytesB));
  CHECK_CUDA(cudaMalloc(&dc, bytesC));

  CHECK_CUDA(cudaMemcpy(da, a, bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, b, bytesB, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dc, 0, bytesC));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // cuBLAS uses column-major storage by default.
  //
  // We want:
  //
  //   C(n x m) = A(n x k) * B(k x m)
  //
  // Therefore:
  //
  //   m_cublas = n
  //   n_cublas = m
  //   k_cublas = k
  //   lda = n
  //   ldb = k
  //   ldc = n
  //
  // with both operands non-transposed.
  const double alpha = 1.0;
  const double beta = 0.0;

  // Warm-up.
  CHECK_CUBLAS(cublasDgemm(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           n,
                           m,
                           k,
                           &alpha,
                           da,
                           n,
                           db,
                           k,
                           &beta,
                           dc,
                           n));

  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;

  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));

  for (int r = 0; r < reps; ++r) {
    CHECK_CUBLAS(cublasDgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             n,
                             m,
                             k,
                             &alpha,
                             da,
                             n,
                             db,
                             k,
                             &beta,
                             dc,
                             n));
  }

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaMemcpy(c, dc, bytesC, cudaMemcpyDeviceToHost));

  int errors = 0;

  for (long long j = 0; j < m_ll; ++j) {
    for (long long i = 0; i < n_ll; ++i) {
      double expected = 0.0;

      for (long long p = 0; p < k_ll; ++p) {
        expected += a[i + p * n_ll] * b[p + j * k_ll];
      }

      double got = c[i + j * n_ll];
      double diff = std::fabs(got - expected);
      double tol = 1.0e-8 + 1.0e-10 * std::fabs(expected);

      if (diff > tol) {
        if (errors < 10) {
          std::fprintf(stderr,
                       "mismatch at (%lld,%lld): got %.17g expected %.17g "
                       "diff %.8e tol %.8e\n",
                       i + 1, j + 1, got, expected, diff, tol);
        }
        ++errors;
      }
    }
  }

  double avgSeconds = (static_cast<double>(ms) / 1.0e3) / reps;

  // 2 FLOPs per multiply-add.
  double flopsPerRep =
      2.0 * static_cast<double>(n_ll) *
      static_cast<double>(m_ll) *
      static_cast<double>(k_ll);

  double gflops = flopsPerRep / avgSeconds / 1.0e9;

  std::printf("cuda_cublas_matmul_2d_f64,%lld,%lld,%d,%.8e,%.8e,%d\n",
              n_ll,
              m_ll,
              reps,
              avgSeconds,
              gflops,
              errors);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  CHECK_CUBLAS(cublasDestroy(handle));

  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(db));
  CHECK_CUDA(cudaFree(dc));

  CHECK_CUDA(cudaFreeHost(a));
  CHECK_CUDA(cudaFreeHost(b));
  CHECK_CUDA(cudaFreeHost(c));

  return errors == 0 ? 0 : 1;
}

