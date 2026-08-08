#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t e, const char *expr, const char *file, int line) {
  if (e == cudaSuccess)
    return;

  std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
               file, line, expr, cudaGetErrorString(e));
  std::abort();
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

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

template <int TILE>
__global__ void matmul_2d_kernel(const float *a, const float *b, float *c,
                                 long long n, long long m, long long k) {
  // Fortran column-major layout:
  //
  //   A(i,p) offset = i + p*n      A shape: n x k
  //   B(p,j) offset = p + j*k      B shape: k x m
  //   C(i,j) offset = i + j*n      C shape: n x m
  //
  // i maps to blockIdx.x/threadIdx.x.
  // j maps to blockIdx.y/threadIdx.y.
  long long i = blockIdx.x * TILE + threadIdx.x;
  long long j = blockIdx.y * TILE + threadIdx.y;

  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  float sum = 0.0f;

  for (long long p0 = 0; p0 < k; p0 += TILE) {
    long long pA = p0 + threadIdx.y;
    long long pB = p0 + threadIdx.x;

    if (i < n && pA < k) {
      As[threadIdx.y][threadIdx.x] = a[i + pA * n];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (pB < k && j < m) {
      Bs[threadIdx.y][threadIdx.x] = b[pB + j * k];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int s = 0; s < TILE; ++s) {
      sum += As[s][threadIdx.x] * Bs[threadIdx.y][s];
    }

    __syncthreads();
  }

  if (i < n && j < m) {
    c[i + j * n] = sum;
  }
}

int main(int argc, char **argv) {
  long long n = parse_i64(argc, argv, 1, 512);
  long long m = parse_i64(argc, argv, 2, 512);
  int reps = parse_i32(argc, argv, 3, 20);

  // Script-compatible choice:
  //
  //   executable n m reps
  //
  // We use k = n by default.
  long long k = n;

  std::size_t elementsA = static_cast<std::size_t>(n * k);
  std::size_t elementsB = static_cast<std::size_t>(k * m);
  std::size_t elementsC = static_cast<std::size_t>(n * m);

  std::size_t bytesA = elementsA * sizeof(float);
  std::size_t bytesB = elementsB * sizeof(float);
  std::size_t bytesC = elementsC * sizeof(float);

  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;

  CHECK(cudaMallocHost(&a, bytesA));
  CHECK(cudaMallocHost(&b, bytesB));
  CHECK(cudaMallocHost(&c, bytesC));

  for (long long p = 0; p < k; ++p) {
    for (long long i = 0; i < n; ++i) {
      long long idx = i + p * n;
      a[idx] = 0.001f * static_cast<float>((i + p) % 100);
    }
  }

  for (long long j = 0; j < m; ++j) {
    for (long long p = 0; p < k; ++p) {
      long long idx = p + j * k;
      b[idx] = 0.002f * static_cast<float>((p + 2 * j) % 100);
    }
  }

  for (std::size_t idx = 0; idx < elementsC; ++idx)
    c[idx] = 0.0f;

  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;

  CHECK(cudaMalloc(&da, bytesA));
  CHECK(cudaMalloc(&db, bytesB));
  CHECK(cudaMalloc(&dc, bytesC));

  CHECK(cudaMemcpy(da, a, bytesA, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b, bytesB, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dc, 0, bytesC));

  constexpr int TILE = 16;

  dim3 block(TILE, TILE, 1);
  dim3 grid(static_cast<unsigned>((n + TILE - 1) / TILE),
            static_cast<unsigned>((m + TILE - 1) / TILE), 1);

  matmul_2d_kernel<TILE><<<grid, block>>>(da, db, dc, n, m, k);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;

  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r) {
    matmul_2d_kernel<TILE><<<grid, block>>>(da, db, dc, n, m, k);
  }
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaMemcpy(c, dc, bytesC, cudaMemcpyDeviceToHost));

  int errors = 0;

  for (long long j = 0; j < m; ++j) {
    for (long long i = 0; i < n; ++i) {
      double expected = 0.0;

      for (long long p = 0; p < k; ++p) {
        expected += static_cast<double>(a[i + p * n]) *
                    static_cast<double>(b[p + j * k]);
      }

      float got = c[i + j * n];
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
      2.0 * static_cast<double>(n) * static_cast<double>(m) *
      static_cast<double>(k);

  double gflops = flopsPerRep / avgSeconds / 1.0e9;

  std::printf("cuda_matmul_2d,%lld,%lld,%d,%.8e,%.8e,%d\n", n, m, reps,
              avgSeconds, gflops, errors);

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

