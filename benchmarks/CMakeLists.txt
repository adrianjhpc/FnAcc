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

set(FNACC_NUM_WARPS "1" CACHE STRING
  "Number of Triton warps for FNACC-generated kernels")

set(FNACC_THREADS_PER_WARP "32" CACHE STRING
  "Triton/CUDA threads per warp")

set(FNACC_NUM_STAGES "3" CACHE STRING
  "Triton pipeline stages")

# -----------------------------------------------------------------------------
# Tool discovery
# -----------------------------------------------------------------------------

if(FNACC_ENABLE_CUDA_BASELINES AND NOT NVCC)
  find_program(NVCC nvcc)
endif()

if(FNACC_ENABLE_OPENACC_BASELINES AND NOT NVFORTRAN)
  find_program(NVFORTRAN nvfortran)
endif()

if(FNACC_ENABLE_OPENMP_BASELINES)
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
      "FNACC_NUM_WARPS=${FNACC_NUM_WARPS}"
      "FNACC_THREADS_PER_WARP=${FNACC_THREADS_PER_WARP}"
      "FNACC_NUM_STAGES=${FNACC_NUM_STAGES}"
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

  add_fnacc_benchmark(fnacc-saxpy
    saxpy_kernel.f90
    saxpy_main.f90)

  add_fnacc_benchmark(fnacc-axpby
    axpby_kernel.f90
    axpby_main.f90)

  add_fnacc_benchmark(fnacc-matrix-add-2d
    matrix_add_2d_kernel.f90
    matrix_add_2d_main.f90)
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
  else()
    message(WARNING "OpenMP baselines requested but no Fortran compiler was found")
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

  add_custom_command(
    OUTPUT "${exe}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${outdir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${workdir}"
    COMMAND "${NVFORTRAN}"
      -O3
      -acc
      "-gpu=${OPENACC_GPU_ARCH}"
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
    add_cuda_benchmark(cuda-saxpy saxpy_cuda.cu)
    add_cuda_benchmark(cuda-axpby axpby_cuda.cu)
    add_cuda_benchmark(cuda-matrix-add-2d matrix_add_2d_cuda.cu)
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

