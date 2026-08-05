cmake_minimum_required(VERSION 3.20)

project(FNACCExamples LANGUAGES NONE)

include(CTest)

# -----------------------------------------------------------------------------#
# User configuration
# -----------------------------------------------------------------------------#

set(LLVM_BUILD "" CACHE PATH
  "Path to the llvm-project build directory containing flang-new, fir-opt, tco, clang")

set(TRITON_OPT "" CACHE FILEPATH
  "Path to Triton triton-opt executable")

set(MLIR_TRANSLATE "" CACHE FILEPATH
  "Path to mlir-translate executable from Triton LLVM")

set(LLC "" CACHE FILEPATH
  "Path to llc executable from Triton LLVM")

set(CUDA_LIB_DIR "/usr/local/cuda/lib64" CACHE PATH
  "Path to CUDA driver library directory containing libcuda.so")

set(FNACC_SM "80" CACHE STRING
  "CUDA SM architecture number, e.g. 80, 90")

set(FNACC_FLANG_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/bin/fnacc-flang"
  CACHE FILEPATH "Path to the fnacc-flang wrapper script")

option(FNACC_BUILD_EXAMPLES_BY_DEFAULT
  "Build all FNACC examples as part of the default build target" ON)

option(FNACC_ENABLE_TESTS
  "Register generated examples as CTest tests" ON)

option(FNACC_VERBOSE_WRAPPER
  "Pass --verbose to the fnacc-flang wrapper" OFF)

# -----------------------------------------------------------------------------#
# Validation helpers
# -----------------------------------------------------------------------------#

function(fnacc_require_path var description)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR
      "${description} is required. Please configure with -D${var}=...")
  endif()
endfunction()

function(fnacc_require_file path description)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR
      "${description} does not exist: ${path}")
  endif()
endfunction()

function(fnacc_require_executable path description)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR
      "${description} does not exist: ${path}")
  endif()

  if(NOT IS_EXECUTABLE "${path}")
    message(FATAL_ERROR
      "${description} is not executable: ${path}")
  endif()
endfunction()

fnacc_require_path(LLVM_BUILD "LLVM build directory")
fnacc_require_path(TRITON_OPT "Triton triton-opt")
fnacc_require_path(MLIR_TRANSLATE "Triton LLVM mlir-translate")
fnacc_require_path(LLC "Triton LLVM llc")

set(FNACC_FLANG "${LLVM_BUILD}/bin/flang-new")
set(FNACC_FIROPT "${LLVM_BUILD}/bin/fir-opt")
set(FNACC_TCO "${LLVM_BUILD}/bin/tco")
set(FNACC_CLANG "${LLVM_BUILD}/bin/clang")

fnacc_require_executable("${FNACC_FLANG_WRAPPER}" "FNACC wrapper")
fnacc_require_executable("${FNACC_FLANG}" "flang-new")
fnacc_require_executable("${FNACC_FIROPT}" "fir-opt")
fnacc_require_executable("${FNACC_TCO}" "tco")
fnacc_require_executable("${FNACC_CLANG}" "clang")
fnacc_require_executable("${TRITON_OPT}" "triton-opt")
fnacc_require_executable("${MLIR_TRANSLATE}" "mlir-translate")
fnacc_require_executable("${LLC}" "llc")

if(NOT EXISTS "${CUDA_LIB_DIR}")
  message(FATAL_ERROR "CUDA_LIB_DIR does not exist: ${CUDA_LIB_DIR}")
endif()

message(STATUS "FNACC examples configuration:")
message(STATUS "  LLVM_BUILD:        ${LLVM_BUILD}")
message(STATUS "  flang-new:         ${FNACC_FLANG}")
message(STATUS "  fir-opt:           ${FNACC_FIROPT}")
message(STATUS "  tco:               ${FNACC_TCO}")
message(STATUS "  clang:             ${FNACC_CLANG}")
message(STATUS "  triton-opt:        ${TRITON_OPT}")
message(STATUS "  mlir-translate:    ${MLIR_TRANSLATE}")
message(STATUS "  llc:               ${LLC}")
message(STATUS "  CUDA_LIB_DIR:      ${CUDA_LIB_DIR}")
message(STATUS "  FNACC_SM:          ${FNACC_SM}")
message(STATUS "  wrapper:           ${FNACC_FLANG_WRAPPER}")

# -----------------------------------------------------------------------------#
# FNACC example helper
# -----------------------------------------------------------------------------#

function(add_fnacc_example name)
  set(options)
  set(one_value_args DIRECTORY)
  set(multi_value_args)
  cmake_parse_arguments(ARG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN})

  if(NOT ARG_DIRECTORY)
    message(FATAL_ERROR "add_fnacc_example(${name}) requires DIRECTORY <dir>")
  endif()

  set(example_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_DIRECTORY}")
  set(kernel_src "${example_src_dir}/kernel.f90")
  set(main_src "${example_src_dir}/main.f90")

  fnacc_require_file("${kernel_src}" "Kernel source for ${name}")
  fnacc_require_file("${main_src}" "Main source for ${name}")

  set(example_build_dir "${CMAKE_CURRENT_BINARY_DIR}/examples/${name}")
  set(example_exe "${example_build_dir}/${name}")
  set(example_runner "${example_exe}.run")

  set(base "${name}")
  set(base_path "${example_build_dir}/${base}")

  set(fir_file "${base_path}.fir")
  set(host_fir_file "${base_path}.host.fir")
  set(host_ll_file "${base_path}.host.ll")
  set(host_obj_file "${base_path}.host.o")
  set(ttir_file "${base_path}.kernels.ttir")
  set(json_file "${base_path}.kernels.json")
  set(ttgir_file "${base_path}.kernels.ttgir.mlir")
  set(kernel_llvm_mlir_file "${base_path}.kernels.llvm.mlir")
  set(kernel_ll_file "${base_path}.kernels.ll")
  set(ptx_file "${base_path}.kernels.ptx")
  set(main_obj_file "${base_path}.main.o")

  set(wrapper_args)
  if(FNACC_VERBOSE_WRAPPER)
    list(APPEND wrapper_args --verbose)
  endif()

  add_custom_command(
    OUTPUT
      "${example_exe}"
      "${example_runner}"
    BYPRODUCTS
      "${fir_file}"
      "${host_fir_file}"
      "${host_ll_file}"
      "${host_obj_file}"
      "${ttir_file}"
      "${json_file}"
      "${ttgir_file}"
      "${kernel_llvm_mlir_file}"
      "${kernel_ll_file}"
      "${ptx_file}"
      "${main_obj_file}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${example_build_dir}"
    COMMAND "${CMAKE_COMMAND}" -E env
      "LLVM_BUILD=${LLVM_BUILD}"
      "TRITON_OPT=${TRITON_OPT}"
      "MLIR_TRANSLATE=${MLIR_TRANSLATE}"
      "LLC=${LLC}"
      "CUDA_LIB_DIR=${CUDA_LIB_DIR}"
      "FNACC_SM=${FNACC_SM}"
      "${FNACC_FLANG_WRAPPER}"
        --kernel-src "${kernel_src}"
        --main-src "${main_src}"
        --workdir "${example_build_dir}"
        --base "${base}"
        -o "${example_exe}"
        ${wrapper_args}
    DEPENDS
      "${FNACC_FLANG_WRAPPER}"
      "${kernel_src}"
      "${main_src}"
    COMMENT "Building FNACC example ${name}"
    VERBATIM
  )

  add_custom_target("${name}" DEPENDS "${example_exe}" "${example_runner}")

  set_property(TARGET "${name}" PROPERTY FOLDER "FNACC Examples")

  if(FNACC_ENABLE_TESTS AND BUILD_TESTING)
    add_test(
      NAME "FNACC.${name}"
      COMMAND "${example_runner}"
    )

    set_tests_properties("FNACC.${name}" PROPERTIES
      LABELS "FNACC"
      WORKING_DIRECTORY "${example_build_dir}"
      ENVIRONMENT
        "FNACC_PTX=${ptx_file};FNACC_KERNELS_JSON=${json_file}"
    )
  endif()

  set(FNACC_EXAMPLE_TARGETS
    ${FNACC_EXAMPLE_TARGETS}
    "${name}"
    PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------#
# Examples
# -----------------------------------------------------------------------------#

set(FNACC_EXAMPLE_TARGETS)

add_fnacc_example(vector-add
  DIRECTORY examples/vector-add)

add_fnacc_example(saxpy
  DIRECTORY examples/saxpy)

add_fnacc_example(axpby
  DIRECTORY examples/axpby)

add_fnacc_example(matrix-add-2d
  DIRECTORY examples/matrix-add-2d)

if(FNACC_BUILD_EXAMPLES_BY_DEFAULT)
  add_custom_target(fnacc-examples ALL
    DEPENDS ${FNACC_EXAMPLE_TARGETS})
else()
  add_custom_target(fnacc-examples
    DEPENDS ${FNACC_EXAMPLE_TARGETS})
endif()

# -----------------------------------------------------------------------------#
# Convenience information target
# -----------------------------------------------------------------------------#

add_custom_target(fnacc-print-config
  COMMAND "${CMAKE_COMMAND}" -E echo "LLVM_BUILD=${LLVM_BUILD}"
  COMMAND "${CMAKE_COMMAND}" -E echo "TRITON_OPT=${TRITON_OPT}"
  COMMAND "${CMAKE_COMMAND}" -E echo "MLIR_TRANSLATE=${MLIR_TRANSLATE}"
  COMMAND "${CMAKE_COMMAND}" -E echo "LLC=${LLC}"
  COMMAND "${CMAKE_COMMAND}" -E echo "CUDA_LIB_DIR=${CUDA_LIB_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E echo "FNACC_SM=${FNACC_SM}"
  COMMAND "${CMAKE_COMMAND}" -E echo "FNACC_FLANG_WRAPPER=${FNACC_FLANG_WRAPPER}"
)

