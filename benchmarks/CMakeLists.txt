cmake_minimum_required(VERSION 3.20)

project(FNACCBenchmarks LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

set(FNACC_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH
  "Path to external FnAcc repository root")

set(FNACC_WRAPPER "${FNACC_ROOT}/bin/fnacc-flang" CACHE FILEPATH
  "Path to fnacc-flang wrapper")

set(FNACC_SM "80" CACHE STRING
  "CUDA SM architecture number, e.g. 80, 90")

set(CUDA_ARCH "sm_${FNACC_SM}" CACHE STRING
  "CUDA architecture passed to nvcc, e.g. sm_80")

set(OPENACC_GPU_ARCH "cc${FNACC_SM}" CACHE STRING
  "OpenACC GPU architecture passed to nvfortran, e.g. cc80")

option(FNACC_ENABLE_FNACC_BENCHMARKS
  "Build FNACC benchmarks" ON)

option(FNACC_ENABLE_OPENMP_BASELINES
  "Build OpenMP CPU baselines" ON)

option(FNACC_ENABLE_OPENMP_GPU_BASELINES
  "Build OpenMP GPU offloading baselines" ON)

option(FNACC_ENABLE_OPENACC_BASELINES
  "Build OpenACC baselines with nvfortran" OFF)

option(FNACC_ENABLE_CUDA_BASELINES
  "Build CUDA baselines with nvcc" ON)

option(FNACC_BENCH_BUILD_BY_DEFAULT
  "Build all enabled benchmark targets by default" ON)

set(NVFORTRAN "" CACHE FILEPATH
  "Path to nvfortran for OpenACC baselines")

set(NVCC "" CACHE FILEPATH
  "Path to nvcc for CUDA baselines")

set(OPENMP_GPU_OFFLOAD_FLAGS "-fopenmp-targets=nvptx64-nvidia-cuda" CACHE STRING
  "Compiler flags for OpenMP target offloading (e.g. -fopenmp-targets=nvptx64-nvidia-cuda or -foffload=nvptx-none)")

set(FNACC_NUM_WARPS "1" CACHE STRING
  "Number of Triton warps for FNACC-generated kernels")

set(FNACC_THREADS_PER_WARP "32" CACHE STRING
  "Triton/CUDA threads per warp")

set(FNACC_MATMUL_NUM_WARPS "4" CACHE STRING
  "Number of Triton warps for FNACC matmul benchmarks")

set(FNACC_MATMUL_F64_NUM_WARPS "4" CACHE STRING
  "Number of Triton warps for FNACC f64 matmul benchmarks")

set(FNACC_NUM_STAGES "3" CACHE STRING
  "Triton pipeline stages")

set(FNACC_F64_MATMUL_STRATEGY "reduce" CACHE STRING
  "FNACC f64 matmul strategy: reduce, fma, or dot")

set(OPENACC_NVFORTRAN_FLAGS "-O3 -tp=zen3 -acc" CACHE STRING
  "nvfortran flags for OpenACC baselines")

set(OPENACC_GPU_FLAGS "-gpu=${OPENACC_GPU_ARCH}" CACHE STRING
  "nvfortran GPU flags for OpenACC baselines")


# -----------------------------------------------------------------------------
# Tool discovery
# -----------------------------------------------------------------------------

if(FNACC_ENABLE_CUDA_BASELINES AND NOT NVCC)
  find_program(NVCC nvcc)
endif()

if(FNACC_ENABLE_OPENACC_BASELINES AND NOT NVFORTRAN)
  find_program(NVFORTRAN nvfortran)
endif()

if(FNACC_ENABLE_OPENMP_BASELINES OR FNACC_ENABLE_OPENMP_GPU_BASELINES)
  enable_language(Fortran OPTIONAL)
endif()

# -----------------------------------------------------------------------------
# Common paths
# -----------------------------------------------------------------------------

set(BENCH_COMMON "${CMAKE_CURRENT_LIST_DIR}/common/bench_utils.f90")

if(NOT EXISTS "${BENCH_COMMON}")
  message(FATAL_ERROR "Missing ${BENCH_COMMON}")
endif()

if(FNACC_ENABLE_FNACC_BENCHMARKS AND NOT EXISTS "${FNACC_WRAPPER}")
  message(FATAL_ERROR "FNACC wrapper not found: ${FNACC_WRAPPER}")
endif()

set(ALL_BENCHMARK_TARGETS)

# -----------------------------------------------------------------------------
# FNACC route
# -----------------------------------------------------------------------------

function(add_fnacc_benchmark name kernel main)
  set(workdir "${CMAKE_CURRENT_BINARY_DIR}/fnacc/${name}")
  set(exe "${workdir}/${name}")

  set(effective_fnacc_num_warps "${FNACC_NUM_WARPS}")

  if("${name}" STREQUAL "fnacc-matmul-2d")
    set(effective_fnacc_num_warps "${FNACC_MATMUL_NUM_WARPS}")
  endif()

  if("${name}" STREQUAL "fnacc-matmul-2d-f64")
    set(effective_fnacc_num_warps "${FNACC_MATMUL_F64_NUM_WARPS}")
  endif()

  add_custom_command(
    OUTPUT "${exe}" "${exe}.run"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${workdir}"
    COMMAND "${CMAKE_COMMAND}" -E env
      "LLVM_BUILD=${LLVM_BUILD}"
      "TRITON_OPT=${TRITON_OPT}"
      "MLIR_TRANSLATE=${MLIR_TRANSLATE}"
      "LLC=${LLC}"
      "CUDA_LIB_DIR=${CUDA_LIB_DIR}"
      "FNACC_SM=${FNACC_SM}"
      "FNACC_NUM_WARPS=${effective_fnacc_num_warps}"
      "FNACC_THREADS_PER_WARP=${FNACC_THREADS_PER_WARP}"
      "FNACC_NUM_STAGES=${FNACC_NUM_STAGES}"
      "FNACC_F64_MATMUL_STRATEGY=${FNACC_F64_MATMUL_STRATEGY}"
      "${FNACC_WRAPPER}"
        --kernel-src "${CMAKE_CURRENT_LIST_DIR}/fnacc/${kernel}"
        --main-src "${CMAKE_CURRENT_LIST_DIR}/fnacc/${main}"
        --extra-src "${BENCH_COMMON}"
        --workdir "${workdir}"
        --base "${name}"
        -o "${exe}"
    DEPENDS
      "${FNACC_WRAPPER}"
      "${CMAKE_CURRENT_LIST_DIR}/fnacc/${kernel}"
      "${CMAKE_CURRENT_LIST_DIR}/fnacc/${main}"
      "${BENCH_COMMON}"
    COMMENT "Building FNACC benchmark ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${exe}" "${exe}.run")
  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(FNACC_ENABLE_FNACC_BENCHMARKS)
  add_fnacc_benchmark(fnacc-vector-add
    vector_add_kernel.f90
    vector_add_main.f90)

  add_fnacc_benchmark(fnacc-vector-add-f64
    vector_add_f64_kernel.f90
    vector_add_f64_main.f90)

  add_fnacc_benchmark(fnacc-saxpy
    saxpy_kernel.f90
    saxpy_main.f90)

  add_fnacc_benchmark(fnacc-saxpy-f64
    saxpy_f64_kernel.f90
    saxpy_f64_main.f90)

  add_fnacc_benchmark(fnacc-axpby
    axpby_kernel.f90
    axpby_main.f90)

  add_fnacc_benchmark(fnacc-axpby-f64
    axpby_f64_kernel.f90
    axpby_f64_main.f90)

  add_fnacc_benchmark(fnacc-matrix-add-2d
    matrix_add_2d_kernel.f90
    matrix_add_2d_main.f90)
  
  add_fnacc_benchmark(fnacc-matmul-2d
    matmul_2d_kernel.f90
    matmul_2d_main.f90)

  add_fnacc_benchmark(fnacc-matmul-2d-f64
    matmul_2d_f64_kernel.f90
    matmul_2d_f64_main.f90)

  add_fnacc_benchmark(fnacc-reduction-sum
    reduction_sum_kernel.f90
    reduction_sum_main.f90)

  add_fnacc_benchmark(fnacc-reduction-dot
    reduction_dot_kernel.f90
    reduction_dot_main.f90)

  add_fnacc_benchmark(fnacc-reduction-dot-f64
    reduction_dot_f64_kernel.f90
    reduction_dot_f64_main.f90)

endif()

# -----------------------------------------------------------------------------
# OpenMP route
# -----------------------------------------------------------------------------

function(add_openmp_benchmark name source)
  if(NOT CMAKE_Fortran_COMPILER)
    message(WARNING "No Fortran compiler available; skipping ${name}")
    return()
  endif()

  set(source_path "${CMAKE_CURRENT_LIST_DIR}/openmp/${source}")

  if(NOT EXISTS "${source_path}")
    message(WARNING "OpenMP benchmark source missing; skipping ${name}: ${source_path}")
    return()
  endif()

  set(openmp_outdir "${CMAKE_CURRENT_BINARY_DIR}/openmp")
  set(openmp_moddir "${CMAKE_CURRENT_BINARY_DIR}/openmp/mod/${name}")

  add_executable("${name}"
    "${BENCH_COMMON}"
    "${source_path}")

  target_compile_options("${name}" PRIVATE -O3 -fopenmp)
  target_link_options("${name}" PRIVATE -fopenmp)

  target_include_directories("${name}" PRIVATE "${openmp_moddir}")

  set_target_properties("${name}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${openmp_outdir}"
    Fortran_MODULE_DIRECTORY "${openmp_moddir}")

  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(FNACC_ENABLE_OPENMP_BASELINES)
  if(CMAKE_Fortran_COMPILER)
    add_openmp_benchmark(openmp-vector-add vector_add_openmp.f90)
    add_openmp_benchmark(openmp-saxpy saxpy_openmp.f90)
    add_openmp_benchmark(openmp-axpby axpby_openmp.f90)
    add_openmp_benchmark(openmp-matrix-add-2d matrix_add_2d_openmp.f90)
    add_openmp_benchmark(openmp-matmul-2d matmul_2d_openmp.f90)
  else()
    message(WARNING "OpenMP baselines requested but no Fortran compiler was found")
  endif()
endif()

# -----------------------------------------------------------------------------
# OpenMP GPU route
# -----------------------------------------------------------------------------

function(add_openmp_gpu_benchmark name source)
  if(NOT CMAKE_Fortran_COMPILER)
    message(WARNING "No Fortran compiler available; skipping ${name}")
    return()
  endif()

  # Assuming you save the new GPU files in a directory called 'openmp_gpu'
  set(source_path "${CMAKE_CURRENT_LIST_DIR}/openmp_gpu/${source}")

  if(NOT EXISTS "${source_path}")
    message(WARNING "OpenMP GPU benchmark source missing; skipping ${name}: ${source_path}")
    return()
  endif()

  set(openmp_gpu_outdir "${CMAKE_CURRENT_BINARY_DIR}/openmp_gpu")
  set(openmp_gpu_moddir "${CMAKE_CURRENT_BINARY_DIR}/openmp_gpu/mod/${name}")

  add_executable("${name}"
    "${BENCH_COMMON}"
    "${source_path}")

  # Separate OMP flags into a list if multiple are provided by the user
  separate_arguments(OMP_OFFLOAD_LIST UNIX_COMMAND "${OPENMP_GPU_OFFLOAD_FLAGS}")

  target_compile_options("${name}" PRIVATE -O3 -fopenmp ${OMP_OFFLOAD_LIST})
  target_link_options("${name}" PRIVATE -fopenmp ${OMP_OFFLOAD_LIST})

  target_include_directories("${name}" PRIVATE "${openmp_gpu_moddir}")

  set_target_properties("${name}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${openmp_gpu_outdir}"
    Fortran_MODULE_DIRECTORY "${openmp_gpu_moddir}")

  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(FNACC_ENABLE_OPENMP_GPU_BASELINES)
  if(CMAKE_Fortran_COMPILER)
    add_openmp_gpu_benchmark(openmp-gpu-vector-add vector_add_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-saxpy saxpy_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-axpby axpby_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-matrix-add-2d matrix_add_2d_openmp_gpu.f90)
    add_openmp_gpu_benchmark(openmp-gpu-matmul-2d matmul_2d_openmp.f90)
  else()
    message(WARNING "OpenMP GPU baselines requested but no Fortran compiler was found")
  endif()
endif()

# -----------------------------------------------------------------------------
# OpenACC route
# -----------------------------------------------------------------------------

function(add_openacc_benchmark name source)
  if(NOT NVFORTRAN)
    message(WARNING "nvfortran not found; skipping ${name}")
    return()
  endif()

  set(source_path "${CMAKE_CURRENT_LIST_DIR}/openacc/${source}")

  if(NOT EXISTS "${source_path}")
    message(WARNING "OpenACC benchmark source missing; skipping ${name}: ${source_path}")
    return()
  endif()

  set(outdir "${CMAKE_CURRENT_BINARY_DIR}/openacc")
  set(workdir "${CMAKE_CURRENT_BINARY_DIR}/openacc")
  set(exe "${outdir}/${name}")

  separate_arguments(OPENACC_NVFORTRAN_FLAGS_LIST UNIX_COMMAND "${OPENACC_NVFORTRAN_FLAGS}")
  separate_arguments(OPENACC_GPU_FLAGS_LIST UNIX_COMMAND "${OPENACC_GPU_FLAGS}")

  add_custom_command(
    OUTPUT "${exe}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${outdir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${workdir}"
    COMMAND "${NVFORTRAN}"
      ${OPENACC_NVFORTRAN_FLAGS_LIST}
      ${OPENACC_GPU_FLAGS_LIST}
      "${BENCH_COMMON}"
      "${source_path}"
      -o "${exe}"
    DEPENDS
      "${BENCH_COMMON}"
      "${source_path}"
    WORKING_DIRECTORY "${workdir}"
    COMMENT "Building OpenACC benchmark ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${exe}")

  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(FNACC_ENABLE_OPENACC_BASELINES)
  if(NVFORTRAN)
    add_openacc_benchmark(openacc-vector-add vector_add_openacc.f90)
    add_openacc_benchmark(openacc-saxpy saxpy_openacc.f90)
    add_openacc_benchmark(openacc-axpby axpby_openacc.f90)
    add_openacc_benchmark(openacc-matrix-add-2d matrix_add_2d_openacc.f90)
    add_openacc_benchmark(openacc-matmul-2d matmul_2d_openacc.f90)
  else()
    message(WARNING
      "OpenACC baselines requested but nvfortran was not found. "
      "Configure with -DNVFORTRAN=/path/to/nvfortran")
  endif()
endif()

# -----------------------------------------------------------------------------
# CUDA route
# -----------------------------------------------------------------------------

function(add_cuda_benchmark name source)
  if(NOT NVCC)
    message(WARNING "nvcc not found; skipping ${name}")
    return()
  endif()

  set(outdir "${CMAKE_CURRENT_BINARY_DIR}/cuda")
  set(exe "${outdir}/${name}")

  add_custom_command(
    OUTPUT "${exe}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${outdir}"
    COMMAND "${NVCC}"
      -O3
      "-arch=${CUDA_ARCH}"
      "${CMAKE_CURRENT_LIST_DIR}/cuda/${source}"
      -o "${exe}"
    DEPENDS "${CMAKE_CURRENT_LIST_DIR}/cuda/${source}"
    COMMENT "Building CUDA benchmark ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${exe}")
  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(FNACC_ENABLE_CUDA_BASELINES)
  if(NVCC)
    add_cuda_benchmark(cuda-vector-add vector_add_cuda.cu)
    add_cuda_benchmark(cuda-vector-add-f64 vector_add_f64_cuda.cu)
    add_cuda_benchmark(cuda-saxpy saxpy_cuda.cu)
    add_cuda_benchmark(cuda-saxpy-f64 saxpy_f64_cuda.cu)
    add_cuda_benchmark(cuda-axpby axpby_cuda.cu)
    add_cuda_benchmark(cuda-axpby-f64 axpby_f64_cuda.cu)
    add_cuda_benchmark(cuda-matrix-add-2d matrix_add_2d_cuda.cu)
    add_cuda_benchmark(cuda-matrix-add-2d-f64 matrix_add_2d_f64_cuda.cu)
    add_cuda_benchmark(cuda-matmul-2d matmul_2d_cuda.cu)
    add_cuda_benchmark(cuda-matmul-2d-f64 matmul_2d_f64_cuda.cu)
    add_cuda_benchmark(cuda-reduction-sum reduction_sum_cuda.cu)
    add_cuda_benchmark(cuda-reduction-dot reduction_dot_cuda.cu)
    add_cuda_benchmark(cuda-reduction-dot-f64 reduction_dot_f64_cuda.cu)
  else()
    message(WARNING
      "CUDA baselines requested but nvcc was not found. "
      "Configure with -DNVCC=/path/to/nvcc")
  endif()
endif()

function(add_cuda_cublas_benchmark name source)
  if(NOT NVCC)
    message(WARNING "nvcc not found; skipping ${name}")
    return()
  endif()

  set(outdir "${CMAKE_CURRENT_BINARY_DIR}/cuda")
  set(exe "${outdir}/${name}")

  add_custom_command(
    OUTPUT "${exe}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${outdir}"
    COMMAND "${NVCC}"
      -O3
      "-arch=${CUDA_ARCH}"
      "${CMAKE_CURRENT_LIST_DIR}/cuda/${source}"
      -lcublas
      -o "${exe}"
    DEPENDS "${CMAKE_CURRENT_LIST_DIR}/cuda/${source}"
    COMMENT "Building CUDA cuBLAS benchmark ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${exe}")
  set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "${name}" PARENT_SCOPE)
endfunction()

if(FNACC_ENABLE_CUDA_BASELINES)
  if(NVCC)
    add_cuda_cublas_benchmark(cuda-matmul-2d-cublas matmul_2d_cublas.cu)
    add_cuda_cublas_benchmark(cuda-matmul-2d-cublas-f64 matmul_2d_f64_cublas.cu)
  else()
    message(WARNING
      "CUDA baselines requested but nvcc was not found. "
      "Configure with -DNVCC=/path/to/nvcc")
  endif()
endif()


# -----------------------------------------------------------------------------
# Aggregate target
# -----------------------------------------------------------------------------

if(FNACC_BENCH_BUILD_BY_DEFAULT)
  add_custom_target(fnacc-all-benchmarks ALL
    DEPENDS ${ALL_BENCHMARK_TARGETS})
else()
  add_custom_target(fnacc-all-benchmarks
    DEPENDS ${ALL_BENCHMARK_TARGETS})
endif()

message(STATUS "FNACC benchmark targets: ${ALL_BENCHMARK_TARGETS}")
