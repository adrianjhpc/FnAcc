#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

static void check_cuda(cudaError_t e, const char *expr, const char *file,
                       int line) {
  if (e == cudaSuccess)
    return;

  std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
               file, line, expr, cudaGetErrorString(e));
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

  std::fprintf(stderr, "cuBLAS error at %s:%d while executing %s: %s\n",
               file, line, expr, cublas_status_string(e));
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

static bool parse_tf32_mode(int argc, char **argv, int index) {
  // Default to TF32 because this is usually the high-performance cuBLAS mode
  // on Ampere/Hopper for single-precision GEMM.
  if (argc <= index)
    return true;

  if (std::strcmp(argv[index], "tf32") == 0)
    return true;

  if (std::strcmp(argv[index], "fp32") == 0)
    return false;

  std::fprintf(stderr,
               "Unknown matmul mode '%s'. Expected 'tf32' or 'fp32'.\n",
               argv[index]);
  std::abort();
}

int main(int argc, char **argv) {
  long long n_ll = parse_i64(argc, argv, 1, 512);
  long long m_ll = parse_i64(argc, argv, 2, 512);
  int reps = parse_i32(argc, argv, 3, 20);
  bool use_tf32 = parse_tf32_mode(argc, argv, 4);

  // Script-compatible choice:
  //
  //   executable n m reps [tf32|fp32]
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

  std::size_t bytesA = elementsA * sizeof(float);
  std::size_t bytesB = elementsB * sizeof(float);
  std::size_t bytesC = elementsC * sizeof(float);

  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;

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
      a[idx] = 0.001f * static_cast<float>((i + p) % 100);
    }
  }

  for (long long j = 0; j < m_ll; ++j) {
    for (long long p = 0; p < k_ll; ++p) {
      long long idx = p + j * k_ll;
      b[idx] = 0.002f * static_cast<float>((p + 2 * j) % 100);
    }
  }

  for (std::size_t idx = 0; idx < elementsC; ++idx)
    c[idx] = 0.0f;

  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;

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
  // Therefore the cuBLAS GEMM call is:
  //
  //   m_cublas = n
  //   n_cublas = m
  //   k_cublas = k
  //   lda = n
  //   ldb = k
  //   ldc = n
  //
  // with both operands non-transposed.
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasComputeType_t compute_type =
      use_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  const char *mode_name =
      use_tf32 ? "cuda_cublas_matmul_2d_tf32"
               : "cuda_cublas_matmul_2d_fp32";

  // For reproducibility/control, set the handle math mode too.
  //
  // The compute_type passed to cublasGemmEx is the main control here.
  if (use_tf32) {
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  }

  // Warm-up.
  CHECK_CUBLAS(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            da,
                            CUDA_R_32F,
                            n,
                            db,
                            CUDA_R_32F,
                            k,
                            &beta,
                            dc,
                            CUDA_R_32F,
                            n,
                            compute_type,
                            CUBLAS_GEMM_DEFAULT));

  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;

  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));

  for (int r = 0; r < reps; ++r) {
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              &alpha,
                              da,
                              CUDA_R_32F,
                              n,
                              db,
                              CUDA_R_32F,
                              k,
                              &beta,
                              dc,
                              CUDA_R_32F,
                              n,
                              compute_type,
                              CUBLAS_GEMM_DEFAULT));
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
        expected += static_cast<double>(a[i + p * n_ll]) *
                    static_cast<double>(b[p + j * k_ll]);
      }

      float got = c[i + j * n_ll];
      double diff = std::fabs(static_cast<double>(got) - expected);
      double tol = 1.0e-2 + 1.0e-3 * std::fabs(expected);

      if (diff > tol) {
        if (errors < 10) {
          std::fprintf(stderr,
                       "mismatch at (%lld,%lld): got %f expected %.8e "
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

  std::printf("%s,%lld,%lld,%d,%.8e,%.8e,%d\n",
              mode_name,
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

