# FNACC reduction validation

`reduction-validation.f90` checks sum, dot, product, minimum, and maximum
reductions for `real32` and `real64`. It covers small inputs, tile boundaries,
non-power-of-two inputs, and inputs large enough to require more than one
hierarchical reduction stage.
It also queries the runtime counters and fails unless the grow-only partial and
scratch buffers have each been allocated, grown, and reused.

The test executable is built by CMake; the shell script only runs and validates
that prebuilt executable. Configure the test as a standalone CMake project:

```sh
cmake -S tests/reduction -B build/reduction \
  -DFNACC_REDUCTION_TEST_DRIVER="$PWD/bin/fnacc-flang" \
  -DLLVM_BUILD="$LLVM_BUILD" \
  -DTRITON_OPT="$TRITON_OPT" \
  -DMLIR_TRANSLATE="$MLIR_TRANSLATE" \
  -DLLC="$LLC"
cmake --build build/reduction
ctest --test-dir build/reduction -V
```

It can instead be part of the main FnAcc build by adding this to the root
`CMakeLists.txt`:

```cmake
add_subdirectory(tests/reduction)
```

With that integration, the default CMake build creates
`build/tests/reduction/fnacc-reduction-validation`, and `ctest` runs it. The
target can also be built explicitly with
`cmake --build build --target fnacc-reduction-validation-build`.

For four-warp reduction lowering, configure with:

```sh
cmake -S tests/reduction -B build/reduction \
  -DFNACC_REDUCTION_TEST_DRIVER="$PWD/bin/fnacc-flang" \
  -DLLVM_BUILD="$LLVM_BUILD" \
  -DTRITON_OPT="$TRITON_OPT" \
  -DMLIR_TRANSLATE="$MLIR_TRANSLATE" \
  -DLLC="$LLC" \
  -DFNACC_NUM_WARPS=4
```

Residual `ttg.warp_id` operations are lowered automatically by the driver;
there is no compatibility flag to enable.

The usual compiler-driver environment (`LLVM_BUILD`, `TRITON_OPT`,
`MLIR_TRANSLATE`, and `LLC`) must be configured independently on each system.
To run the already-built executable manually:

```sh
tests/reduction/run-reduction-validation.sh \
  build/tests/reduction/fnacc-reduction-validation
```

Set `FNACC_REDUCTION_STATS=1` for any FNACC program to print a one-line runtime
workspace summary at process exit.
