# FnACC

FnACC is an experimental directive-based programming model for moving
Fortran loop kernels to GPUs and other accelerators. Its source model is
deliberately similar to OpenACC and OpenMP offload, while its compiler path is
built directly into LLVM Flang.

The implemented backend uses [Triton](https://triton-lang.org/) to
generate code for NVIDIA GPUs through CUDA and AMD GPUs through HIP/ROCm.
Kernel recognition, launch planning, metadata, and the host runtime interface
are backend-neutral so that other code-generation backends and accelerator
targets can be added without changing the Fortran programming model.

The FnACC compiler changes currently live on the
[FnACC branch of the LLVM fork](https://github.com/adrianjhpc/llvm-project/tree/FnACC).
The compiler driver lives in a separate repository.

> [!IMPORTANT]
> FnACC is a prototype. The accepted Fortran subset, directive syntax, runtime
> ABI, and generated JSON may change. The compiler source and regression tests
> are the authoritative contract.

## Contents

- [Current capabilities](#current-capabilities)
- [Compilation pipeline](#compilation-pipeline)
- [GPU targets](#gpu-targets)
- [Building the toolchain](#building-the-toolchain)
- [Quick start](#quick-start)
- [Programming model](#programming-model)
- [Directive reference](#directive-reference)
- [Data movement and lifetime](#data-movement-and-lifetime)
- [Supported kernels](#supported-kernels)
- [Expressions and scalar values](#expressions-and-scalar-values)
- [Reductions](#reductions)
- [Types, ranks, and storage](#types-ranks-and-storage)
- [Compilation and linking](#compilation-and-linking)
- [Compiler-driver reference](#compiler-driver-reference)
- [Runtime configuration](#runtime-configuration)
- [Compiler and runtime architecture](#compiler-and-runtime-architecture)
- [Backend artifact contract](#backend-artifact-contract)
- [Performance tuning and profiling](#performance-tuning-and-profiling)
- [Testing](#testing)
- [Extending FnACC](#extending-fnacc)
- [Diagnostics and troubleshooting](#diagnostics-and-troubleshooting)
- [Known limitations](#known-limitations)

## Current capabilities

FnACC currently provides:

- `parallel` lowering for recognised one- and two-dimensional Fortran loops;
- arbitrary runtime loop lower and upper bounds for elementwise, stencil, and
  reduction kernels, with a unit loop step;
- descriptor-based host launches through ABI v3 by default, with explicit v2
  compatibility and variadic array/scalar bindings;
- one or more output assignments in a recognised loop;
- `real(4)`, `real(8)`, and signed `integer(1|2|4|8)` device expressions;
- affine induction-variable expressions, including the common
  integer-to-real conversion used by mesh initialisation loops;
- two-dimensional stencils with halo offsets, mixed-rank coordinate arrays,
  conditional affine indices, and supported fixed nested-loop expansions;
- specialised single- and double-precision two-dimensional matrix
  multiplication;
- one-dimensional sum, dot-product, product, minimum, and maximum reductions;
- fused two-dimensional multi-result reductions using one common reduction
  operator;
- hierarchical, multi-kernel device reductions with reusable workspaces;
- one- and multi-warp Triton reduction lowering;
- persistent data regions with nested ownership;
- `copyin`, `create`, `copyout`, `delete`, `present`, `update`, and `release`
  data operations;
- derived-component data designators such as
  `chunk%tiles(1)%field%density0`;
- `no_copyback` launch behavior for device-resident outputs;
- opt-in asynchronous resident launches with `FNACC_ASYNC_RESIDENT=1`;
- explicit synchronization with `!$fnacc wait`;
- IEEE-default FP32 matmul and explicit `matmul_precision(tf32|tf32x3)`;
- explicit-shape, assumed-shape, pointer/heap-backed, and allocatable arrays
  when their storage is contiguous;
- multiple separately compiled embedded FnACC bundles in one executable;
- initial end-to-end NVIDIA CUDA and AMD HIP/ROCm target support;
- embedded PTX images for CUDA and HSACO images for HIP;
- per-device and per-accelerator-context runtime state; and
- a backend-neutral kernel plan and artifact contract, with Triton as the only
  complete code-generation backend at present and CUDA and HIP as its
  supported accelerator targets.

## Compilation pipeline

The implemented end-to-end paths are:

```text
Fortran + !$fnacc
  -> Flang parse tree and semantics
  -> FIR with fnacc.launch and FnACC data operations
  -> kernel recognition and backend-neutral planning
  -> Triton TTIR -> target-specific TTGIR -> LLVM MLIR -> LLVM IR
       CUDA: NVPTX + CUDA libdevice -> PTX
       HIP:  AMDGPU + OCML/OCKL/control bitcode -> object -> HSACO
  -> embedded typed device-image/JSON bundle
  -> host object + matching FnACC CUDA Driver API or HIP runtime
```

The recogniser is intentionally fail-closed. A `parallel` region is compiled
only when every relevant operation can be represented in the selected kernel
plan. Unsupported work is not silently discarded or left to execute on the
host.

## GPU targets

The code-generation backend and accelerator target are separate choices.
`--fnacc-backend triton` selects the kernel code generator;
`--fnacc-target cuda|hip` selects the device toolchain, image format, and host
runtime.

| Target | Architecture example | Subgroup width | Embedded image | Runtime library |
| --- | --- | --- | --- | --- |
| `cuda` | `sm_90a` | `32` | PTX | `FortranFNACCRuntime` + CUDA Driver API |
| `hip` | `gfx90a`, `gfx942` | `64` for `gfx9*` by default; otherwise `32` | HSACO | `FortranFNACCRuntimeHIP` + `libamdhip64` |

CUDA remains the default target for compatibility. Select AMD explicitly:

```sh
fnacc-flang --fnacc-target hip --fnacc-gpu-arch gfx942 ...
```

`--fnacc-target` also accepts `nvidia`, `amd`, and `rocm` as normalized
spellings. `--fnacc-sm` remains the CUDA compatibility option, while
`--fnacc-amd-arch` is an AMD compatibility spelling for
`--fnacc-target hip --fnacc-gpu-arch ARCH`.

All FnACC-bearing objects linked into one executable must target the same
accelerator platform. The final link invocation must use the same
`--fnacc-target` so that the driver selects the matching runtime library.

## Building the toolchain

### Required tools

The driver needs an FnACC-enabled LLVM/Flang build and a matching Triton/LLVM
lowering toolchain:

```sh
export LLVM_BUILD=/path/to/llvm-project/build
export TRITON_OPT=/path/to/triton-opt
export MLIR_TRANSLATE=/path/to/triton-llvm/build/bin/mlir-translate
export LLC=/path/to/triton-llvm/build/bin/llc
export LLVM_LINK="$LLVM_BUILD/bin/llvm-link"
export OPT="$LLVM_BUILD/bin/opt"
```

For CUDA, make the CUDA Driver API and libdevice installation discoverable:

```sh
# Optional when libcuda is not in the default linker search path.
export CUDA_LIB_DIR=/usr/local/cuda/lib64

# Usually auto-detected; set this only when necessary.
export FNACC_CUDA_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc
```

For HIP, provide a ROCm installation and `ld.lld`:

```sh
export ROCM_PATH=/opt/rocm
export LD_LLD="$ROCM_PATH/llvm/bin/ld.lld"
```

Configure one or both runtime targets:

```sh
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;flang" \
  -DFLANG_FNACC_RUNTIME=ON \
  -DFLANG_FNACC_RUNTIME_BACKEND=BOTH
```

`FLANG_FNACC_RUNTIME_BACKEND` accepts `CUDA`, `HIP`, or `BOTH` and defaults to
`CUDA`. A HIP or `BOTH` build must find `hip/hip_runtime_api.h` and
`libamdhip64`; set `ROCM_PATH`, `HIP_PATH`, `HIP_DRIVER_INCLUDE_DIR`, or
`HIP_DRIVER_LIBRARY` when they are outside normal locations. A CUDA or `BOTH`
build must find the CUDA driver headers and library.

`FLANG_FNACC=ON` remains a compatibility spelling when
`FLANG_FNACC_RUNTIME` is not set explicitly.

Typical rebuild targets are:

```sh
cmake --build "$LLVM_BUILD" \
  --target fir-opt flang FortranFNACCRuntime FortranFNACCRuntimeHIP
```

When only one target was configured, omit the other runtime target from the
build command. These are configuration fragments: preserve the target,
assertion, compiler, and dependency settings of your working LLVM/Triton build.
With an existing Ninja build, the equivalent rebuild is:

```sh
ninja -C "$LLVM_BUILD" fir-opt flang FortranFNACCRuntime
```

After changing device lowering, recompile the affected Fortran source objects
and relink the application. Relinking alone does not regenerate embedded GPU
code. Changing the default host launch ABI also requires recompiling the host
launch sites. Rebuild and relink the matching runtime when its implementation
changes.

NVTX instrumentation is optional and disabled by default in the revised runtime.
A normal runtime build does not require `nvtx3/nvtx3.hpp`. For an instrumented
build, define `FNACC_ENABLE_NVTX=1` on the runtime CMake target, add the directory
containing `nvtx3/` to its include paths, and link `${CMAKE_DL_LIBS}` where needed.
Set include paths in CMake and regenerate the Ninja build; do not edit generated
`build.ninja` rules. Runtime targets also require the platform thread dependency
(for example, CMake `Threads::Threads`).

## Quick start

### Vector addition with a persistent data region

```fortran
module vector_kernels
  implicit none
contains
  subroutine vector_add(n, a, b, c)
    integer, intent(in) :: n
    real, intent(in) :: a(n), b(n)
    real, intent(out) :: c(n)
    integer :: i

    !$fnacc parallel tile(256) no_copyback
    do i = 1, n
      c(i) = a(i) + b(i)
    end do
  end subroutine
end module

program example
  use vector_kernels
  implicit none
  integer, parameter :: n = 1000000
  real, allocatable :: a(:), b(:), c(:)

  allocate(a(n), b(n), c(n))
  a = 1.0
  b = 2.0

  !$fnacc enter data copyin(a, b) create(c)
  !$fnacc present(a, b, c)

  call vector_add(n, a, b, c)

  !$fnacc exit data copyout(c) delete(a, b, c)

  if (any(c /= 3.0)) error stop "validation failed"
end program
```

Compile, link, and run it on NVIDIA CUDA (the default target):

```sh
/path/to/FnAcc/bin/fnacc-flang \
  --fnacc-target cuda --fnacc-gpu-arch sm_90a \
  example.f90 -O3 -o example
FNACC_DEVICE=0 ./example
```

Compile the same source for AMD HIP/ROCm:

```sh
/path/to/FnAcc/bin/fnacc-flang \
  --fnacc-target hip --fnacc-gpu-arch gfx942 \
  example.f90 -O3 -o example
FNACC_DEVICE=0 ./example
```

For separate compilation:

```sh
fnacc-flang -O3 -c kernels.f90 -o kernels.o
fnacc-flang -O3 -c main.f90 -o main.o
fnacc-flang main.o kernels.o -o example
```

FnACC objects are conventional relocatable objects. Each can contain its own
embedded device bundle, and multiple bundles may be linked into the same
executable. For HIP, repeat `--fnacc-target hip` on every FnACC compilation and
on the final link so the driver selects `FortranFNACCRuntimeHIP`.

## Programming model

### Parallel loops

`!$fnacc parallel` applies to the immediately following `do` construct. There
is no matching end directive.

```fortran
!$fnacc parallel tile(128)
do i = lower, upper
  c(i) = a(i) + b(i)
end do
```

For a rank-two kernel, the outer loop normally represents the second Fortran
dimension and the inner loop the first:

```fortran
!$fnacc parallel tile(16, 16)
do k = y_min, y_max
  do j = x_min, x_max
    c(j, k) = alpha * a(j, k) + beta * b(j, k)
  end do
end do
```

### Loop requirements

The general elementwise, stencil, and reduction recognisers require:

- a constant loop step of `1`;
- a recoverable lower bound and trip count;
- one top-level loop for a rank-one kernel;
- one outer and one inner logical loop for a rank-two kernel; and
- array subscripts that the recogniser can prove are derived safely from the
  logical induction variables and supported scalar indices.

Lower and upper bounds may be arbitrary runtime integer expressions. Empty
ranges are represented by a zero trip count.

Matrix multiplication remains more restrictive: all three canonical loop
lower bounds must currently be the constant `1`.

### Logical tile versus hardware block size

`tile(...)` describes the logical number of elements handled by one device
program in each dimension. It is not the CUDA thread-block size or HIP workgroup
size.

The requested warp count is preserved by the updated scheduling analysis rather
than unconditionally reducing ordinary kernels to one effective warp. The default
request remains one warp unless overridden; eight warps is a tuning choice, not
a new global default. Check the driver's `effective warps` message and generated
JSON after rebuilding.

The hardware block size comes from the selected schedule:

```text
threads_per_cta = num_warps * threads_per_warp
```

CUDA currently requires `threads_per_warp=32`. A one-dimensional tile of 1024
with one warp therefore means that 32 CUDA threads cooperate to process 1024
logical elements. HIP accepts subgroup widths of `32` or `64`; the driver
defaults `gfx9*` architectures to wave64 and later architectures to wave32.

Default logical tiles are:

| Kernel | Default logical tile |
| --- | --- |
| 1-D elementwise or reduction | `1024` |
| 2-D elementwise, stencil, or reduction | `16, 16` |
| f32 matrix multiplication | `16, 16, 32` |
| f64 matrix multiplication | `16, 16, 8` |

Use `FNACC_DEBUG=1` to print the grid, logical tile, subgroup, and hardware
block selected for each launch. The generated per-kernel JSON is the
authoritative schedule.

## Directive reference

FnACC directives are case-insensitive. The recommended free-form sentinel is
`!$fnacc`. The frontend also accepts `!@fnacc`; fixed-form sentinel handling
depends on the Flang source form in use. The driver primarily scans
conventional FnACC sentinels when deciding whether to run the accelerator
pipeline.

### `parallel`

```fortran
!$fnacc parallel [tile(...)] [pack(...)] [reduction(...)] [matmul_precision(...)] [no_copyback]
do ...
  ...
end do
```

| Clause | Purpose |
| --- | --- |
| `tile(x[, y[, z]])` | Compile-time logical tile shape. `z` is the reduction/K tile for matrix multiplication. |
| `pack(name:host, ...)` | Explicitly use host-visible temporary-buffer behavior for the named array arguments. |
| `pack(name:device, ...)` | Use or create cached device storage for the named array arguments. |
| `reduction(+:s)` | Additive reduction into `s`. |
| `reduction(*:s)` | Multiplicative reduction into `s`. |
| `reduction(min:s)` | Minimum reduction into `s`. |
| `reduction(max:s)` | Maximum reduction into `s`. |
| `matmul_precision(ieee\|tf32\|tf32x3)` | FP32 matmul input precision; default `ieee`. Explicit reduced-precision modes require a supported CUDA target. |
| `no_copyback` | Do not automatically copy arrays written by this launch back to the host. Keep their results in cached device storage. |

`pack` currently names simple variables, not arbitrary designators or array
sections. Data directives have broader designator support.

### Data and synchronization directives

| Directive | Behavior |
| --- | --- |
| `!$fnacc enter data copyin(a)` | Begin a nested data region, acquire persistent device storage for `a`, and copy its current host value on a new allocation. |
| `!$fnacc enter data create(c)` | Begin a nested data region and acquire persistent storage for `c` without initializing a new device allocation from the host. |
| `!$fnacc present(a, c)` | Assert that every listed object already has a live device allocation. It neither allocates nor transfers data. |
| `!$fnacc update device(a)` | Create or resize persistent storage as required and copy the current host value to the device. |
| `!$fnacc update host(c)` | Copy a present device allocation to the host. Fails when the object is not present. |
| `!$fnacc exit data copyout(c)` | Request a host update when the innermost region is the final owner, then end that region. |
| `!$fnacc exit data delete(a, b, c)` | Mark listed objects as belonging to the innermost region and release that region's ownership on exit. |
| `!$fnacc release(a, b)` | Release unowned persistent allocations. It cannot release an allocation still owned by an active data region. |
| `!$fnacc release all` | Release all persistent allocations when no data region is active. `release_all` is an alias. |
| `!$fnacc wait` | Wait for prior work on the active FnACC context's stream. It performs no transfer. |

An `enter data` directive creates one ownership frame even when it contains
several clauses. Its matching `exit data` ends the innermost frame.

### Continued directives

Use normal free-form directive continuation. Repeat the sentinel on each
continued line and place `&` after the sentinel on continuation lines:

```fortran
!$fnacc enter data &
!$fnacc& copyin(chunk%tiles(1)%field%density0) &
!$fnacc& copyin(chunk%tiles(1)%field%energy0) &
!$fnacc& create(chunk%tiles(1)%field%pressure)
```

Do not continue a directive using an ordinary, non-directive source line.

### Data designators

Data directives accept Fortran variables and derived-component designators,
including component chains with scalar subscripts:

```fortran
!$fnacc enter data &
!$fnacc& copyin(chunk%tiles(1)%field%density0) &
!$fnacc& create(chunk%tiles(1)%field%energy0)

!$fnacc present(chunk%tiles(1)%field%density0)
!$fnacc update host(chunk%tiles(1)%field%energy0)

!$fnacc exit data &
!$fnacc& copyout(chunk%tiles(1)%field%energy0) &
!$fnacc& delete(chunk%tiles(1)%field%density0, &
!$fnacc&        chunk%tiles(1)%field%energy0)
```

Allocatable and pointer components are lowered through their descriptors.
Storage must still be contiguous and sizeable by the data-operation ABI.
General array-section mapping is not yet a supported contract.

## Data movement and lifetime

### Default launch behavior

When an array has no cached allocation and no device placement is requested,
the launch uses temporary device storage:

- read and read/write arrays are copied host to device;
- the kernel runs;
- written host-target arrays are copied device to host; and
- temporary storage is freed.

This is convenient for isolated kernels but expensive for repeated launches.

### Persistent placement

`pack(...:device)`, `enter data`, `create`, and `update device` can establish a
persistent allocation keyed by the host data address in the active accelerator
context.

Once an allocation is present, subsequent launches use it automatically even
when their `parallel` directives omit `pack(...:device)`. This
present-if-cached rule allows data placement to be controlled outside the
kernel procedure.

During a persistent lifetime:

- a cache miss for a read target creates storage and copies host data;
- a cache miss for a write-only target creates uninitialized storage;
- later launches reuse the allocation;
- host writes are invisible until `update device`;
- device writes are invisible on the host until an automatic copyback,
  `update host`, or final-owner `copyout`; and
- the allocation remains local to its CUDA or HIP context.

### `no_copyback`

Use `no_copyback` when the values written by a launch will be consumed by
later device work:

```fortran
!$fnacc parallel tile(16,16) no_copyback
do k = y_min, y_max
  do j = x_min, x_max
    pressure(j,k) = (1.4_8 - 1.0_8) * density(j,k) * energy(j,k)
  end do
end do
```

The clause affects written arrays only. Required inputs are still made
available on the device. Written arrays are treated as device targets, are
retained in the cache, and are not copied back automatically. Use
`update host(pressure)` or a final `copyout(pressure)` before host code reads
the result.

The selected behavior is recorded in kernel JSON as
`"copy_back_writes": false`.

### `present`

`present` is an assertion and has no OpenACC-style fallback allocation:

```fortran
!$fnacc present(density, energy, pressure)
```

It is useful at procedure boundaries to catch a missing enclosing data region.
Every object must already be present in the active accelerator context;
otherwise the runtime reports an error. `present` neither acquires
nested-region ownership nor performs a transfer.

### Nested data regions

Data regions are stack structured. An inner region may acquire new objects or
share objects already owned by an outer region:

```fortran
!$fnacc enter data copyin(a) create(c)

! Outer work may use a and c.

!$fnacc enter data copyin(a, b) create(tmp)
!$fnacc present(a, b, c, tmp)

! Inner work may use a, b, c, and tmp.

!$fnacc exit data copyout(b) delete(a, b, tmp)

! a and c are still controlled by the outer region.

!$fnacc exit data copyout(c) delete(a, c)
```

The runtime reference-counts region ownership:

- acquiring an already present object adds ownership to the inner frame
  without reallocating it;
- `copyout` on an inner reference is deferred when an enclosing owner remains;
- the host copy occurs when the final owning region requests `copyout`;
- exiting the inner region releases only ownership acquired by that frame;
- allocation storage is freed only after the final region reference is
  released; and
- `copyout` and `delete` operands must belong to the innermost region, so an
  inner exit cannot accidentally release an outer-only object.

An allocation cannot be resized while it is owned by a data region.
`release` and `release all` are for allocations outside active region
ownership; close the owning region first.

### Recommended long-lived pattern

```fortran
!$fnacc enter data copyin(a, b) create(c, work)

do step = 1, timesteps
  call kernel_a(n, a, b, work)
  call kernel_b(n, work, c)
end do

!$fnacc exit data copyout(c) delete(a, b, c, work)
```

If host code changes `a` during the region, add:

```fortran
!$fnacc update device(a)
```

before its next device consumer.

### Synchronization

The runtime owns one nonblocking CUDA or HIP stream and one completion event
per accelerator context. Work submitted through the same context is ordered in
that stream.

Use `!$fnacc wait`:

- before host code that needs completion but not a host transfer;
- before external CUDA or HIP work whose stream has no explicit dependency on
  the FnACC stream;
- as a phase boundary before changing device or context ownership; or
- in tests that require completion at a precise source point.

Host updates, final copyouts, reductions that return host scalars, releases,
and host-target launch paths are synchronization points in the current
runtime. `wait` covers only the active FnACC context and its runtime stream; it
does not synchronize unrelated CUDA/HIP contexts or caller-created streams.

## Supported kernels

### Elementwise and multi-output kernels

Rank-one and rank-two loops may contain recognised expressions and one or more
array stores:

```fortran
!$fnacc parallel tile(128)
do i = lower, upper
  sum(i) = a(i) + b(i)
  difference(i) = a(i) - b(i)
end do
```

Both the v3 host descriptor and the retained v2 binding interface support
variadic arguments. They are not restricted to the older
three-input/three-scalar shape. Practical limits instead come from the
recogniser, generated kernel signature, and backend.

Straight-line stores may feed later expressions in the same logical
iteration. Mutable scalar temporaries are accepted only when proven private to
that iteration.

### Stencils and affine indexing

A rank-two loop is recognised as `stencil2d` when its accesses can be reduced
to supported functions of the logical induction variables and captured index
scalars. Supported patterns include:

- constant halo offsets such as `a(j-1,k)`, `a(j+1,k)`, and `a(j,k+1)`;
- arbitrary loop and array lower bounds;
- independent per-array lower bounds and element strides;
- rank-one coordinate arrays projected onto either loop dimension;
- reversed affine indices such as `x_min-j`;
- conditional affine choices for donor/upwind/downwind indices;
- supported `min`/`max` clamps in index expressions;
- conditional stores and multiple output arrays; and
- fixed, compile-time-expandable nested stencil loops such as a small
  `j:j+1`, `k:k+1` neighborhood.

Example:

```fortran
!$fnacc parallel tile(16,16)
do k = y_min, y_max
  do j = x_min, x_max+2
    if (flux(j,k) < 0.0_8) then
      upwind = j+2
      donor = j+1
      downwind = j
    else
      upwind = j-1
      donor = j
      downwind = j+1
    end if

    output(j,k) = field(donor,k) + field(upwind,k) + &
                  field(downwind,k)
  end do
end do
```

FnACC masks the output domain. The program is responsible for declaring and
maintaining sufficient halo storage for every input offset.

Indirect gathers through arbitrary array-valued indices, data-dependent loop
bounds, and loop-carried stencil dependences remain unsupported.

### Matrix multiplication

The matrix-multiplication recogniser accepts the canonical three-loop form:

```fortran
!$fnacc parallel tile(16,16,32)
do j = 1, n
  do i = 1, m
    acc = 0.0
    do p = 1, k
      acc = acc + a(i,p) * b(p,j)
    end do
    c(i,j) = acc
  end do
end do
```

`real(4)` and `real(8)` matrices are supported. The lower bounds of all three
loops must currently be `1`.

FP32 matmul uses `ieee` input precision by default. Masked matmul inputs are
zero-padded so partial tiles do not contribute undefined values. To opt into
reduced-precision tensor-core input arithmetic on supported NVIDIA GPUs:

```fortran
!$fnacc parallel tile(64,64,32) matmul_precision(tf32)
```

| Mode | Behavior |
| --- | --- |
| `ieee` | Default FP32 input precision; no opt-in to TF32 input rounding. |
| `tf32` | Explicit reduced-precision input arithmetic. |
| `tf32x3` | Three-product TF32 decomposition for improved accuracy over TF32; not a guarantee of bitwise IEEE equivalence. |

The clause applies to one recognised FP32 matmul launch. It is rejected on
FP64 matmul, elementwise kernels, and reductions. The implemented HIP backend
rejects TF32 modes. The CUDA driver requires SM80 or newer for TF32 modes;
TF32x3 also requires the corresponding Triton decomposition pass. Custom
TTIR-to-TTGIR pipelines must retain the required decomposition before matmul
acceleration. Validate numerical error for the application's inputs and
acceptance criteria; do not change validation tolerances merely to hide errors.

Select the f64 code-generation strategy with:

```sh
fnacc-flang --fnacc-f64-matmul-strategy reduce ...
fnacc-flang --fnacc-f64-matmul-strategy fma ...
fnacc-flang --fnacc-f64-matmul-strategy dot ...
```

Performance and toolchain compatibility are GPU- and Triton-version
dependent. Validate numerical results and benchmark on the target system.

## Expressions and scalar values

### Floating-point operations

| Category | Supported forms |
| --- | --- |
| Arithmetic | `+`, `-`, `*`, `/`, unary negation, supported integer powers lowered into the expression tree |
| Unary intrinsics | `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh` |
| Min/max | `min`, `max`, and recognised numeric min/max FIR forms |
| Comparisons | `<`, `<=`, `>`, `>=`, `==`, `/=` using ordered floating-point comparisons |
| Logical combination | Recognised boolean combinations used by conditional expressions and stores |
| Selection | `merge` and supported `if`/select expression trees |

### Integer operations

Signed `integer(1)`, `integer(2)`, `integer(4)`, and `integer(8)` expressions
support:

- addition, subtraction, multiplication, and signed division;
- `abs`;
- signed `min` and `max`;
- signed relational comparisons and equality/inequality; and
- selection over supported integer expressions.

### Type conversion

All array element types must be supported, and all output arrays in one kernel
must currently have the same element type. Read arrays of different supported
types can participate where the expression preserves their types—for example,
an integer material or region array used in a condition alongside real output
arrays.

General type-changing `fir.convert` operations in arithmetic expressions are
rejected. A deliberate exception recognises the integer-to-real conversion of
an affine induction-variable expression, for example:

```fortran
vertexx(j) = xmin + dx * float(j - x_min)
```

This distinction avoids silently dropping a real precision conversion while
still supporting common mesh-initialisation code.

### Read-only scalar capture

A scalar that is read but not written by the kernel is loaded on the host and
passed by value, similar to `firstprivate`:

```fortran
c(i) = alpha * a(i) + beta * b(i)
```

Real or integer expression captures must be compatible with the expression
that consumes them. Integer values used only to build affine subscripts are
classified separately as index captures.

### Iteration-private temporaries

FnACC has no source-level `private` clause. A mutable scalar reference is
promoted to device SSA and treated as private to one logical iteration only
when the compiler proves that it:

- is defined within the logical iteration before every use;
- is not read before being written;
- is not loop-carried;
- is not conditionally left undefined; and
- is not observed by host code after the launch.

```fortran
!$fnacc parallel tile(128)
do i = lower, upper
  tmp = alpha * a(i)
  c(i) = tmp + b(i)
end do
```

Unsafe mutable references fail compilation with a diagnostic such as
`mutable scalar reference is neither iteration-private nor a reduction`.

## Reductions

### One-dimensional reductions

FnACC recognises sum, dot product, product, minimum, and maximum patterns:

```fortran
sum = 0.0
!$fnacc parallel tile(256) reduction(+:sum)
do i = lower, upper
  sum = sum + a(i)
end do

dot = 0.0
!$fnacc parallel tile(256) reduction(+:dot)
do i = lower, upper
  dot = dot + a(i) * b(i)
end do

product = 1.0
!$fnacc parallel tile(256) reduction(*:product)
do i = lower, upper
  product = product * a(i)
end do

smallest = huge(smallest)
!$fnacc parallel tile(256) reduction(min:smallest)
do i = lower, upper
  smallest = min(smallest, a(i))
end do
```

The original host scalar participates in the result. An empty range leaves it
unchanged. Choose an initial value suitable for the operator and element type.

### Fused two-dimensional reductions

A two-dimensional traversal may produce several reductions in one launch when
all results have the same type and use one common operator:

```fortran
!$fnacc parallel tile(16,16) &
!$fnacc& reduction(+:volume_sum, +:mass_sum, +:energy_sum)
do k = y_min, y_max
  do j = x_min, x_max
    volume_sum = volume_sum + volume(j,k)
    mass_sum = mass_sum + volume(j,k) * density(j,k)
    energy_sum = energy_sum + volume(j,k) * density(j,k) * energy(j,k)
  end do
end do
```

Fixed neighborhood expansions may be used inside a recognised two-dimensional
reduction when their bounds can be proven and expanded at compile time.

### Hierarchical implementation

The primary kernel writes one partial result per logical device program. A
synthetic `reduction_stage1d` kernel then reduces those partials recursively
until one result remains. Fused reductions use a corresponding variadic result
layout.

Partial and scratch buffers are cached as grow-only workspaces per accelerator
context. Repeated reductions reuse those allocations.

```sh
FNACC_REDUCTION_STATS=1 ./program
```

prints allocation, growth, reuse, capacity, primary-launch, and stage-launch
counters at process exit. These workspace counters cover the original partial
and scratch buffers, not every auxiliary result allocation. `FNACC_DEBUG=1` prints individual stages.

The revised multi-result finalization path enqueues the final reduction stages
and preserves each result in a packed device buffer. The device-stage path uses
one final explicit wait and one host transfer for the combined results rather
than waiting and transferring separately for every result. Original host seeds
still participate in each reduction. Host-returned scalar reductions remain
synchronous even when asynchronous resident array launches are enabled.

## Types, ranks, and storage

| Feature | Supported today |
| --- | --- |
| Device expression elements | `real(4)`, `real(8)`, signed `integer(1\|2\|4\|8)` |
| Elementwise/stencil kernel rank | 1 or 2 |
| Matrix multiplication rank | 2 |
| Reduction rank | 1, plus recognised fused rank-2 reductions |
| Descriptor data operations | Rank 1 through 3 |
| Storage | Contiguous explicit-shape, assumed-shape, pointer/heap-backed, and allocatable arrays/components |
| Runtime extent and index ABI | Signed 32-bit values |

Data-operation lowering has two principal paths:

1. FIR descriptors are lowered to an ABI containing the data pointer, element
   size, rank, extents, and byte strides.
2. Sized scalars and explicit-shape references are lowered to a raw pointer
   and byte count.

The runtime validates descriptor contiguity. Noncontiguous and strided
sections, unsupported ranks, or objects whose size cannot be established are
rejected rather than copied with an invented extent.

Release a cached allocatable before deallocating or reallocating it. The cache
is keyed by the host data address; retaining an entry across host reallocation
would otherwise leave stale identity and size information.

## Compilation and linking

### Driver pipeline

For an FnACC source, `fnacc-flang` performs:

1. a syntax-only Flang invocation to generate module files;
2. FIR emission;
3. the `fnacc-pipeline`, producing host FIR, device IR, and JSON;
4. per-kernel device-IR splitting;
5. TTIR-to-TritonGPU lowering;
6. TritonGPU-to-LLVM-MLIR lowering;
7. LLVM-MLIR-to-LLVM-IR translation;
8. target-specific device-library linking and image generation:
   - CUDA links libdevice when required and emits PTX with NVPTX `llc`;
   - HIP links OCML/OCKL and ROCm control bitcode, emits an AMDGPU object,
     and links HSACO with `ld.lld`;
9. device-image and JSON embedding; and
10. host-object generation followed by a relocatable link.

The result of `-c` is a conventional relocatable object containing host code,
the embedded device bundle, metadata, and its registration constructor. No
sidecar PTX, HSACO, or JSON files are required at run time.

### Data-only FnACC sources

A source containing FnACC data or synchronization directives but no
`parallel` kernel is valid. The driver runs the frontend and host runtime
lowering, detects that the generated kernel list is empty, skips device code
generation and embedding, and emits a host-only FnACC object.

`FNACC_ALLOW_EMPTY_KERNELS=1` is not required for this case. It is only an
escape hatch when a source contains an FnACC `parallel` launch but the pipeline
unexpectedly emits no kernel; the default is to diagnose that inconsistency.

### Ordinary Fortran sources

Sources without a recognised FnACC sentinel are delegated to the configured
Flang driver. `-E`, `-S`, and `-fsyntax-only` are also delegated and do not run
accelerator code generation. Use `--fnacc-force` or `--fnacc-disable` to
override source auto-detection.

### Separate compilation and multiple bundles

Multiple FnACC sources, objects, and archives may contribute embedded bundles
to one executable. The driver assigns a stable per-source bundle key so kernel
IDs remain disjoint across separately compiled inputs, while direct
`fir-opt` tests retain predictable sequential IDs.

The runtime validates bundle registration and diagnoses kernel identity/name
collisions rather than silently choosing one definition.

FnACC-host objects also use normal Flang external procedure ABI names when
calling procedures defined in ordinary Flang objects. Top-level FnACC
definitions expose the corresponding trailing-underscore compatibility entry
point, allowing mixed FnACC/plain object links.

### Runtime linking

When it detects FnACC code, the driver adds the runtime matching
`--fnacc-target`:

```text
cuda: -L$LLVM_BUILD/lib -lFortranFNACCRuntime    -lcuda     -lstdc++
hip:  -L$LLVM_BUILD/lib -lFortranFNACCRuntimeHIP -lamdhip64 -lstdc++
```

with appropriate runtime search paths. Object and archive inputs are scanned
for FnACC symbols. Use `--fnacc-runtime` when FnACC code is visible only
through `-lNAME` and cannot be detected from a named input file. The link
target must match the target used to compile every embedded bundle.

## Compiler-driver reference

### Traditional options

The wrapper accepts normal compile/link options including `-c`, `-o`, `-I`,
`-J`, `-L`, `-l`, `-D`, `-U`, `-O`, `-g`, `-f...`, `-m...`, `-Wl,...`, and `--`.
Flags are routed to the relevant frontend, host-codegen, or final-link stage.

### FnACC options

| Option | Meaning |
| --- | --- |
| `--fnacc-force` | Run FnACC lowering for every Fortran source. |
| `--fnacc-disable` | Delegate every source to Flang. |
| `--fnacc-runtime` | Force FnACC runtime libraries into the final link. |
| `--fnacc-no-runtime` | Do not add the runtime automatically. |
| `--fnacc-launch-abi N` | Host launch ABI: `3` by default, or explicit `2` compatibility. |
| `--fnacc-target NAME` | Accelerator platform: `cuda` (default) or `hip`. The spellings `nvidia`, `amd`, and `rocm` are normalized. |
| `--fnacc-gpu-arch ARCH` | Target architecture such as `sm_90a`, `gfx90a`, or `gfx942`. |
| `--fnacc-sm N` | CUDA compatibility option accepting `80`, `sm_80`, `cc80`, or `sm_90a`. |
| `--fnacc-amd-arch ARCH` | Compatibility spelling that selects HIP and an AMD `gfx...` architecture. |
| `--fnacc-backend NAME` | Preferred backend; default `auto`. |
| `--fnacc-fallback-backend NAME` | Backend used when the preferred backend rejects a kernel; default `triton`. |
| `--fnacc-backend-fallback` | Enable fallback; currently the default. |
| `--fnacc-no-backend-fallback` | Fail instead of using the fallback backend. |
| `--fnacc-num-warps N` | Requested warps per CTA; a power of two and at most 32. |
| `--fnacc-threads-per-warp N` | Subgroup width. CUDA requires `32`; HIP accepts `32` or `64`. |
| `--fnacc-num-stages N` | Triton pipeline stages, currently 1 through 16. |
| `--fnacc-f64-matmul-strategy NAME` | `reduce`, `fma`, or `dot`. |
| `--fnacc-cuda-lib-dir DIR` | CUDA Driver API library directory. |
| `--fnacc-cuda-libdevice FILE` | CUDA `libdevice.10.bc`; normally auto-detected. |
| `--fnacc-rocm-path DIR` | ROCm installation prefix; default `ROCM_PATH` or `/opt/rocm`. |
| `--fnacc-rocm-device-lib-dir DIR` | Directory containing ROCm device bitcode such as `ocml.bc` and `ockl.bc`. |
| `--fnacc-hip-lib-dir DIR` | Directory containing `libamdhip64`. |
| `--fnacc-workdir DIR` | Parent directory for a unique intermediate tree. |
| `--fnacc-keep` | Keep intermediate files. |
| `--fnacc-verbose` | Print commands before executing them. |
| `--fnacc-dry-run` | Print commands without executing them. |
| `--fnacc-stop-after STAGE` | Stop one FnACC source after an internal stage. |

Compatibility aliases without the `fnacc-` prefix are accepted for schedule,
work-directory, verbosity, and stop controls.

Useful stop stages include `modgen`, `fir`, `fnacc-pipeline`, `ttgir`,
`llvm-mlir`, `llvm-ir`, `ptx`, `hsaco`, `device-image`, `embed`, `host-ll`,
`host-obj`, `object`, `objects`, and `link`.

```sh
fnacc-flang --fnacc-verbose --fnacc-keep \
  --fnacc-stop-after fnacc-pipeline -c kernels.f90
```

### Driver environment variables

| Variable | Purpose |
| --- | --- |
| `LLVM_BUILD` | FnACC-enabled LLVM/Flang build directory. |
| `TRITON_OPT` | `triton-opt` executable. |
| `MLIR_TRANSLATE` | Matching `mlir-translate`. |
| `LLC` | Matching `llc` used to emit PTX or an AMDGPU object. |
| `LLVM_LINK` | Matching `llvm-link`; defaults to `$LLVM_BUILD/bin/llvm-link`. |
| `OPT` | Matching `opt`; defaults to `$LLVM_BUILD/bin/opt`. |
| `LD_LLD` | `ld.lld` used to link HSACO; defaults to `$ROCM_PATH/llvm/bin/ld.lld`. |
| `FLANG`, `FIROPT`, `TCO`, `CLANG` | Optional tool overrides. |
| `FLANG_INTRINSIC_MODULES_PATH` | Override Flang's intrinsic-module directory. |
| `CUDA_LIB_DIR` | CUDA Driver API library directory. |
| `FNACC_CUDA_LIBDEVICE` | CUDA `libdevice.10.bc` override. The driver auto-detects it when generated IR references `__nv_*`. |
| `ROCM_PATH` | ROCm installation prefix; default `/opt/rocm`. |
| `FNACC_ROCM_DEVICE_LIB_DIR` | ROCm bitcode directory containing OCML/OCKL and control bitcode. |
| `FNACC_HIP_LIB_DIR` | Directory containing `libamdhip64`; defaults to `$ROCM_PATH/lib` or `lib64`. |
| `FNACC_LAUNCH_ABI` | Default host launch ABI; `3`. An explicit `--fnacc-launch-abi` overrides it. |
| `FNACC_TARGET` | Default accelerator target: `cuda` or `hip`. |
| `FNACC_GPU_ARCH` | Default target architecture such as `sm_90a` or `gfx942`. |
| `FNACC_SM` | CUDA architecture compatibility variable. |
| `FNACC_AMD_GPU_ARCH` | AMD architecture compatibility variable; default `gfx90a` when HIP is selected and no architecture is supplied. |
| `FNACC_BACKEND` | Preferred backend; default `auto`. |
| `FNACC_FALLBACK_BACKEND` | Fallback backend; default `triton`. |
| `FNACC_ALLOW_BACKEND_FALLBACK` | Boolean backend-fallback control. |
| `FNACC_NUM_WARPS` | Requested warp count; current default `1`. |
| `FNACC_THREADS_PER_WARP` | Subgroup width; current default `32`. |
| `FNACC_NUM_STAGES` | Pipeline stages; current default `3`. |
| `FNACC_F64_MATMUL_STRATEGY` | Default f64 matmul strategy. |
| `FNACC_WORKDIR` | Intermediate-directory parent. |
| `FNACC_TTIR_TO_TTGIR_PASSES` | Advanced TTIR-to-TTGIR pass-pipeline override. |
| `FNACC_TTGIR_TO_LLVM_PASSES` | Advanced TTGIR-to-LLVM-MLIR pass-pipeline override. |
| `FNACC_HIP_TTIR_TO_TTGIR_PASSES` | HIP-only TTIR-to-TritonGPU pass-pipeline override. |
| `FNACC_HIP_TTGIR_TO_LLVM_PASSES` | HIP-only TritonGPU-to-LLVM-MLIR pass-pipeline override. |
| `FNACC_ALLOW_EMPTY_KERNELS` | Permit an empty kernel list despite an FnACC `parallel` launch; default false. Data-only sources do not need it. |

The work directory must be writable and contain no whitespace because some
toolchain components and generated command lines require whitespace-free
intermediate paths.

## Runtime configuration

| Variable | Purpose |
| --- | --- |
| `FNACC_DEVICE` | Device ordinal for either runtime. Takes precedence over the vendor-specific variable; default `0`. |
| `FNACC_CUDA_DEVICE` | CUDA device ordinal when `FNACC_DEVICE` is unset. |
| `FNACC_HIP_DEVICE` | HIP device ordinal when `FNACC_DEVICE` is unset. |
| `FNACC_USE_CURRENT_CONTEXT` | Use the caller's current CUDA or HIP context instead of retaining a primary context. |
| `FNACC_ASYNC_RESIDENT` | Set to `1` to enqueue eligible cached-array launches without waiting after each launch; unset/default retains synchronous completion. |
| `FNACC_DEBUG` | Print initialization, bundle, cache, data-region, launch, grid, tile, ABI, and reduction diagnostics. |
| `FNACC_REDUCTION_STATS` | Print aggregate reduction-workspace counters at exit. |
| `FNACC_MATMUL_SHARED_BYTES` | Advanced f32 matmul dynamic-shared-memory override; cannot be below the computed safe minimum. |
| `FNACC_MATMUL_F64_SHARED_BYTES` | Advanced f64 matmul dynamic-shared-memory override; cannot be below the computed safe minimum. |
| `FNACC_PTX_DIR`, `FNACC_PTX`, `FNACC_KERNELS_JSON` | Legacy CUDA external-bundle debugging fallbacks. Driver-built objects normally use embedded images and JSON. |

### Asynchronous resident execution

```sh
FNACC_ASYNC_RESIDENT=1 FNACC_DEVICE=0 ./program
```

This is independent of the host launch ABI: selecting v3 does not enable async
execution. Only eligible array launches whose allocations remain cached may
return before device completion. Temporary/host-output paths and reductions
returning host scalars retain synchronization. Host transfers and allocation
lifetime operations order against queued work on the FnACC stream.

Use explicit waits around a timed sequence:

```fortran
! Warm up before measuring.
call compute()
!$fnacc wait

t0 = wall_time()
do r = 1, reps
  call compute()
end do
!$fnacc wait
t1 = wall_time()

! Fetch results outside the timed interval if measuring resident compute.
!$fnacc update host(c)
```

Without the second wait, the timer may measure submission rather than completed
GPU work. Without the first, warm-up work may leak into the timed interval.
An eventual successful validation does not prove that the timed interval
included device execution. State whether transfers are included in any reported
performance measurement.

### Accelerator context behavior

The CUDA build uses the CUDA Driver API; the HIP build uses an internal adapter
over the HIP runtime/module APIs. By default each initializes its platform,
selects `FNACC_DEVICE` or the vendor-specific device variable, retains the
device's primary context, and restores the caller's previous current context on
return.

With `FNACC_USE_CURRENT_CONTEXT=1`, the caller must make a context for the
selected platform current before entering FnACC. Runtime state is created for
that exact context and the runtime does not retain or release it. Using the
option with no current context is a fatal error.

Modules, function handles, device allocations, data-region frames, streams,
events, and reduction workspaces are stored per context and are never reused in
another context or accelerator runtime.

### Thread safety

The revised runtime uses thread-local active-context/operation state, a short
registry lock, per-context mutexes, and a shared lifetime guard. Operations on
one context remain serialized; operations on separate initialized contexts can
proceed independently. Cold initialization and cleanup require broader locking.
Callers must still coordinate shared data-region lifetimes and their own host
accesses. This does not introduce multiple user queues within a context.

Debug, async-resident, and reduction-stat flags are sampled once per outer
runtime operation. Immutable kernel metadata and resolved descriptors are reused
to reduce repeated parsing and lookup; live argument values, layouts, and
ownership remain validated. Device/context selection is not permanently cached
across calls.

## Compiler and runtime architecture

### Principal implementation areas

| Area | Principal files |
| --- | --- |
| Parse tree and syntax | `flang/include/flang/Parser/parse-tree-fnacc.h`, `flang/lib/Parser/fnacc-parsers.cpp` |
| Unparsing and semantics | `flang/lib/Parser/unparse.cpp`, `flang/lib/Semantics/resolve-names.cpp` |
| PFT and FIR generation | `flang/include/flang/Lower/PFTBuilder.h`, `flang/lib/Lower/PFTBuilder.cpp`, `flang/lib/Lower/Bridge.cpp` |
| FnACC dialect | `flang/include/flang/Optimizer/Dialect/FNACC/FNACCOps.td`, `FNACCDialect.td`, `flang/lib/Optimizer/Dialect/FNACC/FNACCDialect.cpp` |
| Recognition and planning | `FNACCKernelAnalysis.h/.cpp`, `FNACCKernelPlan.h` |
| Triton backend | `flang/lib/Optimizer/Dialect/FNACC/FNACCLowerToTriton.cpp` |
| Host runtime lowering | `flang/lib/Optimizer/Dialect/FNACC/FNACCLowerToRuntime.cpp` |
| External ABI aliases | `flang/lib/Optimizer/Dialect/FNACC/FNACCEmitFortranAliases.cpp` |
| Pass pipeline | `FNACCPasses.td`, `FNACCPipelines.cpp` |
| CUDA/HIP runtimes | `flang/lib/Runtime/FNACC/fnacc_runtime.cpp`, built as `FortranFNACCRuntime` or `FortranFNACCRuntimeHIP` |
| Compiler wrapper | `FnAcc/bin/fnacc-flang` |

The FIR dialect includes:

- `fnacc.launch` with tile sizes, pack targets, reduction metadata, and
  `no_copyback` behavior;
- `fnacc.terminator` for its single-block region;
- `fnacc.data_region_enter` and `fnacc.data_region_exit`;
- `fnacc.copyin`, `fnacc.create`, `fnacc.copyout`, and `fnacc.delete`;
- `fnacc.present`, `fnacc.update_host`, and `fnacc.update_device`;
- `fnacc.release`, `fnacc.release_all`, and `fnacc.wait`.

Hand-written MLIR tests must terminate `fnacc.launch` with
`fnacc.terminator`; `fir.end` is not the FnACC region terminator.

### Pass pipeline

The registered `fnacc-pipeline` performs:

1. `fnacc-assign-kernel-ids` for stable per-bundle IDs and symbols;
2. recognition, planning, backend selection, device IR, and JSON emission;
3. `fnacc-lower-to-runtime` for host launch and data calls; and
4. optional `fnacc-emit-fortran-aliases` for external Fortran ABI
   compatibility.

`fnacc-outline-kernels` also exists for development experiments but is not a
normal stage of the driver route.

### Recognition and consumed operations

Recognition builds an `ElementwiseKernel` containing loop and extent sources,
array arguments and accesses, scalar/index captures, output expressions,
reduction metadata, and a list of consumed FIR operations.

Validation requires every operation with observable effects in the launch
region to be represented by the kernel or accepted as structural
loop/terminator machinery. When adding a pattern, update consumed-operation
accounting with the recogniser. Otherwise a valid pattern may be rejected—or
an effect could be erased without being represented in device code.

### Host launch ABI v3 (default)

The compiler constructs a host request containing launch dimensions, array
records, scalar/index captures, and reduction-result records, then emits:

```text
__fnacc_launch_v3(request_pointer)
```

This replaces the generated sequence of begin/bind/commit calls. Scalar values
and reduction seeds are captured for the invocation. The runtime consumes the
request during the call; async mode does not make stack request storage into a
device-owned object. Cached device allocations follow the normal lifetime rules.
The v3 entry reuses the checked launch machinery under one outer operation and
context guard; it does not change the numerical device kernel.

```sh
# Default v3; an environment override can change this.
fnacc-flang -O3 -c kernel.f90

# Explicit selection overrides FNACC_LAUNCH_ABI.
fnacc-flang --fnacc-launch-abi 3 -O3 -c kernel.f90
fnacc-flang --fnacc-launch-abi 2 -O3 -c kernel.f90
```

Both `fnacc-pipeline` and standalone `fnacc-lower-to-runtime` default to v3:

```sh
fir-opt --fnacc-pipeline="launch-abi=2 ttir-output=k.ttir json-output=k.json" input.fir
fir-opt --fnacc-lower-to-runtime="launch-abi=2" input.fir
```

V3 lowering requires an explicit supported 64-bit `x86_64` or `aarch64` host
`llvm.target_triple`. Missing triples, x32/ILP32 forms, and other host targets
are rejected by the current implementation. Use v2 for unsupported host targets;
hand-written MLIR intended to test v3 must declare its actual supported target.

### V2 compatibility and device metadata

V2 remains available through `__fnacc_begin_launch_v2`, typed
`__fnacc_bind_*_v2` calls, and `__fnacc_commit_launch_v2`. Keep these runtime
symbols and their compatibility tests.

**The host launch selection and JSON device launch ABI are different contracts.**
The v3 integration retains device JSON `launch_abi_version = 2`, device signatures,
and backend-private argument accounting. Do not rename JSON fields, change their
version to 3, or change runtime device-ABI checks just because v3 is the host
default. No new device precision mode is implied by v3.

Maintain the default consistently in the driver, `FNACCPipelines.cpp`, and
`FNACCPasses.td`. Rebuild generated pass declarations through the normal build;
do not edit generated files. The driver logs the selected host ABI and includes
it in its toolchain fingerprint.

## Backend artifact contract

Kernel recognition and scheduling are backend-neutral. `FNACCKernelPlan`
contains stable identity, recognised expressions/accesses, tile and subgroup
schedule, public ABI, pack bindings, copyback policy, and optional synthetic
reduction-stage plans.

`FNACCCodegenBackend` provides:

- a backend name;
- emitted device-IR and runtime image kinds;
- per-plan support queries and rejection reasons;
- module framing and kernel emission; and
- the count of backend-private pointer arguments.

The only complete code-generation backend currently registered is `triton`.
It can emit for the `cuda` or `hip` accelerator target. Backend selection
supports `auto`, a named preferred backend, an optional fallback, and detailed
fallback diagnostics. Mixed code-generation backends or accelerator targets
within one emitted device module are not supported.

### JSON metadata

Schema version 1 and backend-contract version 1 include top-level backend,
accelerator-target, and image fields plus a list of kernel descriptors. Each
descriptor records:

- stable bundle-qualified `id` and `name`;
- backend, accelerator target, device-IR kind, and device-image kind;
- image index and file;
- kernel kind and rank;
- logical tile, warp/stage schedule, subgroup width, and threads per CTA;
- device launch ABI version (still `2` with the v3 host launcher);
- array, scalar, output, and reduction-result counts;
- parameter roles, slots, names, types, array dimensions, and layout fields;
- loop lower-bound and extent roles;
- pack bindings and `copy_back_writes`;
- backend-private pointer count; and
- reduction operator and synthetic-stage identity where applicable.

Legacy PTX and Triton-private field aliases remain while older tools are being
retired.

### Runtime images

The typed registration ABI accepts PTX, cubin, and HSACO images. The normal
Triton CUDA path emits PTX; the Triton HIP path emits an AMDGPU object and links
it into HSACO. The CUDA runtime rejects HIP/HSACO metadata, and the HIP runtime
rejects CUDA/PTX or cubin metadata, preventing an image from being loaded by
the wrong platform runtime.

A future direct-PTX, CUDA Tile IR, or other backend may reuse the kernel plan,
public ABI, metadata, embedding, and runtime dispatch layers, but its driver
must still produce an image supported by the selected runtime.

Adding a compiler backend enum alone is insufficient. Driver dispatch,
manifest/embed logic, runtime loading, contract validation, and tests must be
updated together.

## Performance tuning and profiling

### Preserve residency across application phases

For an iterative application, establish one intended data lifetime across setup
and the timestep loop. Fetch only the fields needed by host consumers and release
storage after its final use. Closing a region at the end of initialization and
opening another at the start of hydro can introduce a large copyout/re-upload.
One MPI process removes inter-process communication, but physical-boundary halo
updates and any application-level tile exchanges may still run.

### Tune individual kernels

Tile shape and warp count are separate parameters. The first Fortran dimension
is contiguous, so wider tiles in that dimension are useful candidates. The
emitted TTGIR layout, array strides, masks, and final device code determine the
actual memory behavior. More warps are not automatically better.

For the measured H100 CloverLeaf long case, the confirmed configuration for
`ideal_gas_kernel.f90`, `reset_field_kernel.f90` (both loops), and
`revert_kernel.f90` is:

```fortran
!$fnacc parallel tile(256,1) no_copyback
```

with `--fnacc-num-warps 8 --fnacc-threads-per-warp 32`. Keep the original ideal-gas
arithmetic. This is an application-specific tuning result, not a replacement for
all stencil or reduction tiles. Earlier interior `64,4` and halo-strip changes
remain separate choices for other kernels. Halo strips should place the long
dimension along the boundary traversal and the short dimension along halo depth;
preserve each loop's bounds and corner dependencies when changing them.

The paired long-run profiles showed the following accumulated device durations:

| Kernel family | Previous FnACC | Tuned FnACC (`256,1`, 8 warps) | Matching earlier CUDA capture | Matching earlier OpenACC capture |
| --- | ---: | ---: | ---: | ---: |
| Ideal gas | 9.279 s | 7.845 s | 7.966 s | 8.827 s |
| Reset, both loops | 8.529 s | 7.427 s | 7.412 s | 7.487 s |
| Revert | 4.437 s | 3.749 s | 3.664 s | 3.792 s |
| All GPU kernels | 193.675 s | 190.446 s | 200.252 s | 198.255 s |

These are sums of kernel durations from particular captures, not application
wall times or a portable speedup claim. The tuned routines account for about
3.224 seconds of the 3.229-second change; other kernels were essentially
unchanged. Full input files, build revisions, clocks, and run conditions are
needed to reproduce a comparison. Validate each configuration and compare repeated
unprofiled runs before accepting small differences.

V3 reduced some host overhead in the microbenchmarks but did not measurably
change CloverLeaf wall time or reported correctness. Do not attribute device
kernel improvements from tile tuning to the host ABI change. Large kernels can
hide host submission savings; small resident kernels are more sensitive to them.

### Inspect generated kernels

Keep intermediates for the exact source/configuration being measured:

```sh
mkdir -p fnacc-inspect
fnacc-flang --fnacc-target cuda --fnacc-gpu-arch sm_90a \
  --fnacc-num-warps 8 --fnacc-threads-per-warp 32 \
  --fnacc-keep --fnacc-workdir "$PWD/fnacc-inspect" \
  -O3 -c reset_field_kernel.f90 -o reset_field_kernel.o

rg --files fnacc-inspect | rg '\.ptx$'
rg -n --glob '*.ptx' '\.entry' fnacc-inspect
rg -n --glob '*.json' '"tile"|"num_warps"|"threads_per_cta"' fnacc-inspect
```

Retain normal application include/module flags and relink before measuring.
The normal driver log does not list every kernel name. Generated `.kernels.json`
and per-kernel filenames map source bundles to names; `.entry` identifies PTX
entry points. Do not assume an old numeric kernel name identifies a new build.

Inspect `.ttgir.mlir` for `sizePerThread`, `threadsPerWarp`, and `warpsPerCTA`;
inspect PTX for address calculation, predicated memory operations, and arithmetic.
Scalar 64-bit loads/stores are not inherently uncoalesced. PTX register names
are virtual registers and cannot establish final physical register usage,
spilling, or achieved occupancy.

### Nsight Systems summaries

For CUDA tracing, use the application's normal single-process invocation and
working directory, with a unique report name:

```sh
FNACC_ASYNC_RESIDENT=1 nsys profile \
  --trace=cuda,nvtx --sample=none \
  -o cloverleaf-fnacc ./clover_leaf

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  cloverleaf-fnacc.nsys-rep > cloverleaf-fnacc-summary.txt
```

NVTX ranges appear only if instrumentation is enabled; CUDA tracing does not
require application NVTX instrumentation. Nsight Systems can also generate these
summaries from its SQLite export. Retain the full trace when investigating
ordering or gaps: aggregate summaries do not locate copies within application
phases or establish overlap.

Compare matching mesh dimensions, step counts, summary frequency, and validation
results. Compare kernel families as well as individual kernels because one
implementation may split work into more launches. Time inside
`cuEventSynchronize`, stream waits, or synchronous copies includes waiting for
GPU work; do not add it to GPU kernel time as independent overhead.

### Nsight Compute and restricted profiling environments

When GPU counter access is available, collect a few invocations of one kernel:

```sh
# Replace ENTRY_NAME with the exact entry from the current generated PTX.
FNACC_ASYNC_RESIDENT=1 ncu \
  --kernel-name-base function --kernel-name ENTRY_NAME \
  --launch-skip 10 --launch-count 3 --kill yes \
  --section LaunchStats --section Occupancy \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --export kernel-profile ./clover_leaf

ncu --import kernel-profile.ncu-rep --page details > kernel-profile.txt
ncu --import kernel-profile.ncu-rep --page raw --csv > kernel-profile.csv
```

Skip/count apply to matching kernel launches, not all application launches.
Counter collection may replay each invocation. `--kill yes` stops the diagnostic
run after collection, so run numerical validation separately to completion.

If `ERR_NVGPUCTRPERM` appears, collection is blocked by platform permissions;
root inside a container is not sufficient by itself. Ask the platform
administrator about supported profiling access. If access cannot be granted,
continue with Nsight Systems durations and generated TTGIR/PTX. Do not treat a
failed counter capture as an occupancy or bandwidth measurement, and stop the
application manually if it continues after the profiling error.

## Testing

### Flang regression suite

Run all Flang tests with:

```sh
cmake --build "$LLVM_BUILD" --target check-flang
```

Run a focused test or directory with:

```sh
"$LLVM_BUILD/bin/llvm-lit" -sv \
  /path/to/llvm-project/flang/test/Lower/FNACC/fnacc-pipeline.f90

"$LLVM_BUILD/bin/llvm-lit" -sv \
  /path/to/llvm-project/flang/test/Lower/FNACC
```

The suite covers parser/unparser behavior, semantics, FIR lowering,
runtime-call lowering, TTIR and JSON, arbitrary loop bounds, affine indices,
mixed-rank stencils, conditional indices, fixed stencil expansions,
multi-output expressions, integer and floating-point operations, assumed-shape
and allocatable descriptors, matmul variants, one- and two-dimensional
reductions, multi-warp lowering, nested data regions, derived-component data
designators, `present`, `no_copyback`, data-only sources, external ABI aliases,
CUDA/HIP accelerator-target JSON and image metadata, and negative diagnostics.

Prefer `CHECK-LABEL`, `CHECK-NEXT`, bounded `CHECK`, and `CHECK-DAG` patterns to
fragile `CHECK-SAME` assertions when operations are intentionally printed on
separate lines. Tests should verify semantics and types rather than local SSA
names.

### Host-ABI regression tests

Keep positive v2 begin/bind/commit checks paired with an explicit
`launch-abi=2` in the command that produces their host IR. Include continued
`RUN` lines when editing test commands. V3 emits one launch call per source
launch, not one replacement for each old begin/bind/commit call.

The dedicated v3 test should cover:

- omitted `launch-abi` selecting v3;
- explicit v3 through the pipeline and standalone lowering;
- explicit v2 compatibility;
- absence of generated v2 begin/bind/commit calls in v3 output;
- request fields, scalar kinds/seeds, and multi-result ordering;
- identical device TTIR/JSON across host ABI selections where expected; and
- unsupported host triples and invalid ABI selections.

Keep checks for `fnacc.launch` IR, data transfers, release, and synchronization
operations unchanged unless their actual semantics change. V2 matmul argument
checks cannot be migrated by merely renaming the callee: v3 passes a request
pointer, and the corresponding request stores must be checked instead.

Run both FnACC test directories:

```sh
"$LLVM_BUILD/bin/llvm-lit" -sv \
  /path/to/llvm-project/flang/test/FNACC \
  /path/to/llvm-project/flang/test/Lower/FNACC
```

Files ending in `.before-v3-default`, `.before-v2-pin`, or `.bak` are editor
backups, not normal `.f90`/`.mlir` lit tests. Review changes and remove backups
before committing. A switch in defaults does not establish that every legacy
test has been migrated; run the suite and inspect remaining failures.

### Reduction validation executable

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

Configure with `-DFNACC_NUM_WARPS=4` to exercise multi-warp reductions.

### CUDA and HIP smoke tests

After building the selected runtime, compile and run a small numerical kernel
on each available target before testing a full application:

```sh
# NVIDIA
fnacc-flang --fnacc-target cuda --fnacc-gpu-arch sm_90a \
  vector_add.f90 -O3 -o vector_add.cuda
FNACC_DEBUG=1 FNACC_DEVICE=0 ./vector_add.cuda

# AMD
fnacc-flang --fnacc-target hip --fnacc-gpu-arch gfx942 \
  vector_add.f90 -O3 -o vector_add.hip
FNACC_DEBUG=1 FNACC_DEVICE=0 ./vector_add.hip
```

Use an architecture that exactly matches the installed GPU. The HIP path is
particularly sensitive to compatible Triton, LLVM, ROCm device-library, and
`ld.lld` revisions.

### Repeated BabelStream measurements

`tools/fnacc-babelstream-stats.py` runs warm-ups and repeated measured trials,
reports bandwidth statistics and robust outliers, and can preserve JSON:

```sh
tools/fnacc-babelstream-stats.py \
  --warmups 1 --runs 9 \
  --arraysize 33554432 --numtimes 100 \
  --json fnacc-babelstream.json \
  ./BabelStream.fnacc.FnACCArray
```

Use the same device visibility, device ordinal, clocks, array size, iteration
count, and warm-up policy for every implementation being compared. A
validation error is a failed benchmark regardless of reported bandwidth.

### Runtime debugging

```sh
fnacc-flang --fnacc-keep --fnacc-verbose \
  --fnacc-target TARGET --fnacc-gpu-arch ARCH -c kernel.f90
FNACC_DEBUG=1 FNACC_DEVICE=0 ./program
```

For CUDA memory checking:

```sh
CUDA_LAUNCH_BLOCKING=1 compute-sanitizer \
  --tool memcheck --error-exitcode 99 ./program
```

Validate results separately from timed launches, and keep required host
updates outside the timed region.

## Extending FnACC

### Add or change directive syntax

1. Add parse-tree nodes in `parse-tree-fnacc.h`.
2. Add parsers in `fnacc-parsers.cpp` and executable-construct routing when
   needed.
3. Resolve contained variables and names in `resolve-names.cpp`.
4. Add unparse support in `unparse.cpp`.
5. Add PFT/Bridge lowering to an existing or new FnACC FIR operation.
6. Define and verify the operation in `FNACCOps.td` and `FNACCDialect.cpp`.
7. Lower it to the runtime or consume it in kernel planning.
8. Add parser, unparser, semantics, FIR, runtime-lowering, and negative tests.

Use `Fortran::lower::SomeExpr` for semantic expressions obtained from
`Fortran::semantics::GetExpr`, and call Bridge expression-address helpers using
their current signature rather than an obsolete location-first overload.

### Add an expression operation

1. Extend `ElementwiseExprKind`.
2. Recognise the exact FIR operation or Flang lowering idiom in
   `FNACCKernelAnalysis.cpp`.
3. Preserve result-kind and element-type semantics, especially conversions.
4. Mark all represented operations as consumed.
5. Emit the expression in every applicable backend.
6. Add rank-one/rank-two, type, edge-case, and negative tests.

Flang may lower one intrinsic differently for different kinds or optimization
levels. Integer `abs`, min/max, `merge`, and real conversion are examples where
matching one apparent FIR spelling is too fragile.

### Add a kernel pattern

1. Define a kernel kind and recognition result.
2. Prove bounds, extents, access ranks/layouts, types, side-effect rules, and
   scalar classification.
3. Construct the backend-neutral schedule and public ABI.
4. Add runtime binding when the host request/binding interfaces cannot represent
   the pattern.
5. Implement backend emission and `querySupport` checks.
6. Emit and validate JSON metadata.
7. Add end-to-end numerical tests and FIR/TTIR/JSON regression tests.

Fail closed. A precise compile-time diagnostic is preferable to a kernel whose
indexing, mutation, or ABI has not been proven safe.

### Add a device backend

1. Implement `FNACCCodegenBackend` against `FNACCKernelPlan`.
2. Give unsupported plans precise `querySupport` diagnostics.
3. Register preferred, automatic, and fallback selection.
4. Emit schema-v1/backend-contract-v1 metadata.
5. Add driver dispatch for the emitted device-IR and image kinds.
6. Extend typed bundle generation if a new runtime image type is required.
7. Extend runtime validation and module loading.
8. Test preferred, automatic, fallback, disabled-fallback, unsupported, and
   mixed-backend cases.

Do not place backend-private parameters in the stable public ABI. Record them
through the private-argument contract.

### Add an accelerator target

An accelerator target is distinct from a code-generation backend. Adding one
requires coordinated changes to:

1. the compiler target option and JSON `accelerator_target` contract;
2. the backend's target-specific image kind and lowering configuration;
3. driver architecture parsing, device-library linking, image generation, and
   final runtime selection;
4. typed bundle image-kind registration;
5. module, allocation, copy, stream/event, and launch operations in the target
   runtime; and
6. target/image mismatch diagnostics and end-to-end device tests.

Do not allow a runtime to accept metadata or images for a different target.

### Change a runtime ABI

Update these together:

- runtime-call creation in `FNACCLowerToRuntime.cpp`;
- exported runtime function signatures;
- JSON parameter roles and types when the device contract also changes;
- driver image/ABI validation;
- compatibility aliases or a schema/ABI version; and
- MLIR lowering and executable integration tests.

Accelerator-owned objects must remain in state keyed by the active CUDA or HIP
context; never infer ownership from a process-global device-pointer cache.

## Diagnostics and troubleshooting

### `FNACC cannot plan launch`

The launch is outside the recognised subset. Read the final recogniser reason
first. Common causes include:

- a non-unit loop step;
- a matrix loop whose lower bound is not `1`;
- an unsupported call, conversion, or intrinsic;
- an indirect or unprovable array subscript;
- a loop-carried dependence;
- an unsupported mutable scalar;
- a descriptor or rank the ABI cannot represent; or
- an operation with observable effects that was not consumed by the plan.

### `FNACC backend selection failed`

The preferred backend is unregistered or its `querySupport` rejected the plan.
Use `--fnacc-backend triton`, permit a Triton fallback, or inspect the detailed
rejection. `--fnacc-no-backend-fallback` is useful in tests that must prove a
specific backend handled every kernel.

### Driver fails at `fnacc-pipeline`

Re-run with `--fnacc-verbose --fnacc-keep`, execute the printed `fir-opt`
command directly, and inspect the retained `.fir`, `.kernels.ttir`,
`.kernels.json`, and `.host.fir` files.

### PTX reports an unresolved `__nv_*` function

CUDA kernels that use operations such as double-precision square root may
reference CUDA libdevice. The driver detects these references, links
`libdevice.10.bc` with `llvm-link`, runs `opt -O3` so required definitions are
materialized/inlined, and only then invokes NVPTX `llc`.

If `ptxas` still reports an unresolved symbol such as `__nv_sqrt`, ensure
`LLVM_LINK`, `OPT`, and `LLC` come from compatible LLVM builds and that the
selected libdevice is compatible with them. Override discovery with
`--fnacc-cuda-libdevice FILE` or `FNACC_CUDA_LIBDEVICE`, retain intermediates,
and run `ptxas` on the generated PTX directly.

### HIP/ROCm toolchain errors

For `--fnacc-target hip`, confirm that:

- `--fnacc-gpu-arch` names the installed GPU, for example `gfx90a` or `gfx942`;
- `ROCM_PATH` points to the intended ROCm installation;
- `LLC`, `MLIR_TRANSLATE`, and `LD_LLD` are compatible with the Triton build;
- the device-library directory contains `ocml.bc`, `ockl.bc`, the required
  `oclc_*` control modules, and an ISA module for the selected `gfx...`; and
- the final link also uses `--fnacc-target hip` and can find `libamdhip64`.

Override device-library discovery with `--fnacc-rocm-device-lib-dir` and HIP
runtime discovery with `--fnacc-hip-lib-dir`. If Triton's AMD pass spellings
differ from the defaults, set `FNACC_HIP_TTIR_TO_TTGIR_PASSES` and/or
`FNACC_HIP_TTGIR_TO_LLVM_PASSES`.

An error that the AMDGPU object or HSACO was not generated usually indicates a
target-architecture or LLVM/ROCm version mismatch. Retain intermediates and run
the printed `llc` and `ld.lld` commands directly.

### Runtime rejects the accelerator target or image kind

CUDA objects carry `accelerator_target=cuda` with PTX/cubin images; HIP objects
carry `accelerator_target=hip` with HSACO images. A target/image mismatch means
the object was linked against the wrong FnACC runtime or bundles for different
targets were mixed. Recompile consistently and repeat the same
`--fnacc-target` on the final link.

### `no kernels were emitted`

Data-only FnACC sources are accepted automatically. If the message says that
an FnACC `parallel` launch was present, recognition or pipeline output is
inconsistent. Inspect retained FIR/JSON rather than setting
`FNACC_ALLOW_EMPTY_KERNELS=1` in normal builds; the variable suppresses a
safety check and does not make a missing kernel execute.

### Undefined `_QP...` references when linking plain objects

Rebuild the FnACC compiler and recompile the affected FnACC object with the
current external-alias pass enabled. FnACC declarations should call ordinary
top-level Flang procedures through `name_`, while FnACC definitions provide a
compatible external entry point. Use `nm` to confirm that callers and
definitions agree.

### `present` or `update host` reports no allocation

The object was never entered into persistent storage, was released by its
owning data region, belongs to another accelerator context, or was used only
through a host-temporary launch. Establish storage with `enter data`, `create`,
`update device`, or `pack(...:device)` before asserting presence or updating
the host.

### `data_delete requires an active ENTER DATA region`

A cached allocation is not proof of an active ownership frame: `update device`
can create a cache entry even if the intended `enter data` never executed.
Do not use `exit data copyout(...)` solely to fetch results if a later procedure
will perform `exit data delete(...)`; the first exit already ends the frame.
Use `update host(...)` for the intermediate fetch and one final region exit.

An earlier frontend bug omitted standalone FnACC directives from the PFT lexical
successor chain. In particular, `enter data create(...)` after an allocation
error check containing `STOP` could be skipped. The fix classifies
`FnACCStandaloneConstruct` as an executable directive in `PFTBuilder.h`.
Rebuild the frontend and affected Fortran objects; moving the directive to a
different procedure is a workaround, not the intended requirement. Verify the
runtime enter/create calls in lowered FIR when diagnosing an old build.

This specific fix does not establish that every unstructured control-flow corner
case is supported; keep the separate termination-shape diagnostics below.

### Data-region ownership errors

`copyout` and `delete` on `exit data` refer to the innermost active frame. Make
sure that frame acquired every listed object. Do not use `release` or
`release all` to bypass a live frame; exit nested regions in last-in,
first-out order.

### Descriptor sizing or contiguity errors

Prefer explicit-shape arrays or supported contiguous descriptors. A `create`
operation must determine the full byte size. Noncontiguous sections and
unsupported ranks cannot be mapped safely by the current runtime ABI.

### FIR verification after control-flow termination

Some PFT/control-flow shapes can still expose a standalone FnACC directive
after a block already terminated by an infinite `do`/conditional `exit`
sequence, producing an error such as:

```text
operation with block successors must terminate its parent block
```

Until that frontend control-flow case is fixed, put the corresponding
`exit data` cleanup on the actual termination path immediately before the
`exit`, or restructure the loop so the directive is reached through an
ordinary fall-through block.

### Wrong results with persistent placement

Check the lifetime in this order:

1. Was each input initialized with `copyin` or `update device`?
2. Did host code modify it after the last host-to-device transfer?
3. Did `no_copyback` intentionally keep the result on the device?
4. Are in-place read/write arguments bound to the same cached object?
5. Was a host update or final-owner `copyout` performed before validation?
6. Did an inner data region defer copyout because an outer owner remained?
7. Was the allocation released only after its final consumer?

Enable `FNACC_DEBUG=1` and inspect context, bundle, cache, region depth,
ownership count, pack targets, byte counts, extents, lower bounds, strides,
grid, tile, subgroup/block size, accelerator target, and image metadata.

## Known limitations

- FnACC is experimental and deliberately recogniser-based rather than a
  general-purpose Fortran device compiler.
- Loop steps must be `1`; matmul loop lower bounds must also be `1`.
- Kernel computation supports logical ranks one and two. Data-descriptor
  operations support ranks one through three.
- Device extents, lower bounds, strides, and index captures use signed 32-bit
  runtime values.
- General type-changing arithmetic conversions are unsupported; the affine
  integer-index-to-real case is handled explicitly.
- All output arrays in one kernel must currently have the same element type.
- Fused multi-result reductions require one common operator and result type.
- Matrix multiplication supports only f32 and f64.
- Triton is the only complete code-generation backend. Its CUDA path emits PTX
  and its HIP path emits HSACO. Cubin is accepted by the CUDA typed-image ABI
  but is not the normal Triton output.
- CUDA and HIP use separate runtime libraries. Mixing CUDA and HIP bundles in
  one executable or loading an image through the wrong runtime is unsupported.
- Mixed code-generation backends or accelerator targets in one device module
  are unsupported.
- The runtime owns one stream per CUDA or HIP context and serializes operations
  within each context; initialization/cleanup also use shared registry/lifetime
  coordination.
- Host ABI v3 currently requires an explicit supported 64-bit x86_64 or aarch64
  target triple. V2 remains the explicit compatibility path.
- The HIP path depends on revision-compatible Triton, LLVM, ROCm device
  libraries, and `ld.lld`; AMD lowering pass names may require the documented
  environment overrides for a particular Triton revision.
- There is no source-level `private` clause; only proven iteration-private
  scalar temporaries are promoted automatically.
- There are no asynchronous queue IDs, user stream clauses, exposed events,
  or cross-stream dependency clauses.
- Noncontiguous sections, arbitrary indirect gathers, complex values,
  character values, derived-type device elements, unsigned arithmetic,
  arbitrary function calls, recursion, and general unstructured control flow
  are outside the current kernel subset.
- Derived-component data objects are supported, but arbitrary array-section
  mapping is not.
- Floating-point min/max, comparisons, transcendental functions, contraction,
  and reduction order inherit backend behavior. Validate NaNs, infinities,
  signed zero, and reproducibility when an application depends on them.
- A frontend control-flow corner case remains for standalone data directives
  reached after certain terminated `do`/`exit` block shapes.
