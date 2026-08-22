cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(FNACCReductionValidation LANGUAGES NONE)
endif()

include(CTest)

get_filename_component(_FNACC_DEFAULT_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(FNACC_ROOT "${_FNACC_DEFAULT_ROOT}" CACHE PATH
  "Path to the FnAcc repository root")
unset(_FNACC_DEFAULT_ROOT)

set(_FNACC_REDUCTION_TEST_DRIVER_DEFAULT "${FNACC_ROOT}/bin/fnacc-flang")
if(DEFINED FNACC_DRIVER AND NOT "${FNACC_DRIVER}" STREQUAL "")
  set(_FNACC_REDUCTION_TEST_DRIVER_DEFAULT "${FNACC_DRIVER}")
elseif(DEFINED ENV{FNACC_DRIVER} AND NOT "$ENV{FNACC_DRIVER}" STREQUAL "")
  set(_FNACC_REDUCTION_TEST_DRIVER_DEFAULT "$ENV{FNACC_DRIVER}")
endif()

set(FNACC_REDUCTION_TEST_DRIVER
  "${_FNACC_REDUCTION_TEST_DRIVER_DEFAULT}" CACHE FILEPATH
  "Path to the fnacc-flang compiler driver")
unset(_FNACC_REDUCTION_TEST_DRIVER_DEFAULT)

set(LLVM_BUILD "$ENV{LLVM_BUILD}" CACHE PATH
  "llvm-project build directory used by fnacc-flang")
set(TRITON_OPT "$ENV{TRITON_OPT}" CACHE FILEPATH
  "Path to triton-opt used by fnacc-flang")
set(MLIR_TRANSLATE "$ENV{MLIR_TRANSLATE}" CACHE FILEPATH
  "Path to mlir-translate used by fnacc-flang")
set(LLC "$ENV{LLC}" CACHE FILEPATH
  "Path to llc used by fnacc-flang")
set(CUDA_LIB_DIR "$ENV{CUDA_LIB_DIR}" CACHE PATH
  "Optional CUDA driver library directory")

set(FNACC_SM "80" CACHE STRING
  "CUDA SM architecture number, e.g. 80 or 90")
set(FNACC_NUM_WARPS "1" CACHE STRING
  "Number of Triton warps for the reduction validation kernels")
set(FNACC_THREADS_PER_WARP "32" CACHE STRING
  "Threads per Triton warp")
set(FNACC_NUM_STAGES "3" CACHE STRING
  "Number of Triton pipeline stages")
set(FNACC_REDUCTION_TEST_COMPILE_FLAGS "-O3" CACHE STRING
  "Additional flags passed to fnacc-flang")

option(FNACC_REDUCTION_TEST_BUILD_BY_DEFAULT
  "Build the reduction validation executable as part of the default build" ON)
option(FNACC_REDUCTION_TEST_KEEP_INTERMEDIATES
  "Keep fnacc-flang intermediate files for the reduction test" OFF)
option(FNACC_REDUCTION_TEST_VERBOSE
  "Print commands run internally by fnacc-flang" OFF)

if(NOT EXISTS "${FNACC_REDUCTION_TEST_DRIVER}")
  message(FATAL_ERROR
    "fnacc-flang was not found at ${FNACC_REDUCTION_TEST_DRIVER}. "
    "Set FNACC_REDUCTION_TEST_DRIVER or FNACC_DRIVER.")
endif()

if(NOT IS_DIRECTORY "${LLVM_BUILD}")
  message(FATAL_ERROR
    "LLVM_BUILD is not a directory: ${LLVM_BUILD}. "
    "Configure with -DLLVM_BUILD=/path/to/llvm-project/build.")
endif()

foreach(tool_var IN ITEMS TRITON_OPT MLIR_TRANSLATE LLC)
  if(NOT EXISTS "${${tool_var}}")
    message(FATAL_ERROR
      "${tool_var} was not found at ${${tool_var}}. "
      "Set -D${tool_var}=/path/to/tool.")
  endif()
endforeach()

if(NOT "${CUDA_LIB_DIR}" STREQUAL "" AND NOT IS_DIRECTORY "${CUDA_LIB_DIR}")
  message(FATAL_ERROR "CUDA_LIB_DIR is not a directory: ${CUDA_LIB_DIR}")
endif()

if("${CMAKE_CURRENT_BINARY_DIR}" MATCHES " ")
  message(FATAL_ERROR
    "The FNACC reduction test build directory must not contain whitespace.")
endif()

separate_arguments(_FNACC_REDUCTION_TEST_COMPILE_FLAGS
  UNIX_COMMAND "${FNACC_REDUCTION_TEST_COMPILE_FLAGS}")

set(_FNACC_REDUCTION_TEST_CONTROL_FLAGS)
if(FNACC_REDUCTION_TEST_KEEP_INTERMEDIATES)
  list(APPEND _FNACC_REDUCTION_TEST_CONTROL_FLAGS --fnacc-keep)
endif()
if(FNACC_REDUCTION_TEST_VERBOSE)
  list(APPEND _FNACC_REDUCTION_TEST_CONTROL_FLAGS --fnacc-verbose)
endif()

set(_FNACC_REDUCTION_TEST_SOURCE
  "${CMAKE_CURRENT_LIST_DIR}/reduction-validation.f90")
set(FNACC_REDUCTION_TEST_EXECUTABLE
  "${CMAKE_CURRENT_BINARY_DIR}/fnacc-reduction-validation")
set(_FNACC_REDUCTION_TEST_INTERMEDIATES
  "${CMAKE_CURRENT_BINARY_DIR}/intermediates")
set(_FNACC_REDUCTION_TEST_MODULES
  "${CMAKE_CURRENT_BINARY_DIR}/modules")
set(_FNACC_REDUCTION_TEST_CONFIG
  "${CMAKE_CURRENT_BINARY_DIR}/fnacc-reduction-validation.config")

string(CONCAT _FNACC_REDUCTION_TEST_CONFIG_CONTENT
  "driver=${FNACC_REDUCTION_TEST_DRIVER}\n"
  "sm=${FNACC_SM}\n"
  "num_warps=${FNACC_NUM_WARPS}\n"
  "threads_per_warp=${FNACC_THREADS_PER_WARP}\n"
  "num_stages=${FNACC_NUM_STAGES}\n"
  "compile_flags=${FNACC_REDUCTION_TEST_COMPILE_FLAGS}\n")
file(GENERATE OUTPUT "${_FNACC_REDUCTION_TEST_CONFIG}"
  CONTENT "${_FNACC_REDUCTION_TEST_CONFIG_CONTENT}")
unset(_FNACC_REDUCTION_TEST_CONFIG_CONTENT)

set(_FNACC_REDUCTION_TEST_ENVIRONMENT
  "LLVM_BUILD=${LLVM_BUILD}"
  "TRITON_OPT=${TRITON_OPT}"
  "MLIR_TRANSLATE=${MLIR_TRANSLATE}"
  "LLC=${LLC}"
  "CUDA_LIB_DIR=${CUDA_LIB_DIR}")

add_custom_command(
  OUTPUT "${FNACC_REDUCTION_TEST_EXECUTABLE}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory
    "${_FNACC_REDUCTION_TEST_INTERMEDIATES}"
    "${_FNACC_REDUCTION_TEST_MODULES}"
  COMMAND "${CMAKE_COMMAND}" -E env
    ${_FNACC_REDUCTION_TEST_ENVIRONMENT}
    "${FNACC_REDUCTION_TEST_DRIVER}"
      ${_FNACC_REDUCTION_TEST_CONTROL_FLAGS}
      --fnacc-workdir "${_FNACC_REDUCTION_TEST_INTERMEDIATES}"
      --fnacc-sm "${FNACC_SM}"
      --fnacc-num-warps "${FNACC_NUM_WARPS}"
      --fnacc-threads-per-warp "${FNACC_THREADS_PER_WARP}"
      --fnacc-num-stages "${FNACC_NUM_STAGES}"
      ${_FNACC_REDUCTION_TEST_COMPILE_FLAGS}
      -I "${_FNACC_REDUCTION_TEST_MODULES}"
      -J "${_FNACC_REDUCTION_TEST_MODULES}"
      "${_FNACC_REDUCTION_TEST_SOURCE}"
      -o "${FNACC_REDUCTION_TEST_EXECUTABLE}"
  DEPENDS
    "${FNACC_REDUCTION_TEST_DRIVER}"
    "${_FNACC_REDUCTION_TEST_SOURCE}"
    "${_FNACC_REDUCTION_TEST_CONFIG}"
  COMMENT "Building FNACC reduction validation executable"
  VERBATIM
)

if(FNACC_REDUCTION_TEST_BUILD_BY_DEFAULT)
  add_custom_target(fnacc-reduction-validation-build ALL
    DEPENDS "${FNACC_REDUCTION_TEST_EXECUTABLE}")
else()
  add_custom_target(fnacc-reduction-validation-build
    DEPENDS "${FNACC_REDUCTION_TEST_EXECUTABLE}")
endif()

if(BUILD_TESTING)
  add_test(NAME fnacc-reduction-validation
    COMMAND "${CMAKE_CURRENT_LIST_DIR}/run-reduction-validation.sh"
      "${FNACC_REDUCTION_TEST_EXECUTABLE}")
endif()
