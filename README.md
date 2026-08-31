# FnACC

FnACC is an experimental directive-based programming model for moving
Fortran loop kernels to GPUs and other accelerators. Its source model is
deliberately similar to OpenACC and OpenMP offload, while its compiler path is
built directly into LLVM Flang.

The current production backend uses [Triton](https://triton-lang.org/) to
generate NVIDIA GPU code. Kernel recognition, launch planning, metadata, and
the host runtime interface are backend-neutral so that other code-generation
backends can be added without changing the Fortran programming model.

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
- [Testing](#testing)
- [Extending FnACC](#extending-fnacc)
- [Diagnostics and troubleshooting](#diagnostics-and-troubleshooting)
- [Known limitations](#known-limitations)

## Current capabilities

FnACC currently provides:

- `parallel` lowering for recognised one- and two-dimensional Fortran loops;
- arbitrary runtime loop lower and upper bounds for elementwise, stencil, and
  reduction kernels, with a unit loop step;
- variadic array and scalar bindings through the version 2 launch ABI;
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
- explicit synchronization with `!$fnacc wait`;
- explicit-shape, assumed-shape, pointer/heap-backed, and allocatable arrays
  when their storage is contiguous;
- multiple separately compiled embedded FnACC bundles in one executable;
- per-device and per-CUDA-context runtime state; and
- a backend-neutral kernel plan and artifact contract, with Triton as the only
  complete code-generation backend at present.

## Compilation pipeline

The implemented end-to-end path is:

```text
Fortran + !$fnacc
  -> Flang parse tree and semantics
  -> FIR with fnacc.launch and FnACC data operations
  -> kernel recognition and backend-neutral planning
  -> Triton TTIR -> TTGIR -> LLVM MLIR -> LLVM IR -> PTX
  -> embedded typed device-image/JSON bundle
  -> host object + FnACC CUDA Driver API runtime
```

The recogniser is intentionally fail-closed. A `parallel` region is compiled
only when every relevant operation can be represented in the selected kernel
plan. Unsupported work is not silently discarded or left to execute on the
host.

## Building the toolchain

### Required tools

The driver needs an FnACC-enabled LLVM/Flang build and a matching Triton
lowering toolchain:

```sh
export LLVM_BUILD=/path/to/llvm-project/build
export TRITON_OPT=/path/to/triton-opt
export MLIR_TRANSLATE=/path/to/triton-llvm/build/bin/mlir-translate
export LLC=/path/to/triton-llvm/build/bin/llc

# Optional when libcuda is not in the default linker search path.
export CUDA_LIB_DIR=/usr/local/cuda/lib64
```

Configure the LLVM tree with the FnACC CUDA runtime enabled:

```sh
cmake -S llvm -B build -DFLANG_FNACC_RUNTIME=ON
```

Add the option to the rest of the LLVM/Flang configuration used for your
build.

`FLANG_FNACC=ON` remains a compatibility spelling when
`FLANG_FNACC_RUNTIME` is not set explicitly.

Typical rebuild targets are:

```sh
cmake --build "$LLVM_BUILD" \
  --target fir-opt flang FortranFNACCRuntime
```

The CUDA driver headers and library must be discoverable when the runtime is
enabled.

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

Compile, link, and run it with:

```sh
/path/to/FnAcc/bin/fnacc-flang example.f90 -O3 -o example
FNACC_CUDA_DEVICE=0 ./example
```

For separate compilation:

```sh
fnacc-flang -O3 -c kernels.f90 -o kernels.o
fnacc-flang -O3 -c main.f90 -o main.o
fnacc-flang main.o kernels.o -o example
```

FnACC objects are conventional relocatable objects. Each can contain its own
embedded device bundle, and multiple bundles may be linked into the same
executable.

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

### Logical tile versus CUDA block size

`tile(...)` describes the logical number of elements handled by one device
program in each dimension. It is not the CUDA thread-block size.

The CUDA block size comes from the selected schedule:

```text
cuda_threads_per_cta = num_warps * threads_per_warp
```

CUDA currently requires `threads_per_warp=32`. A one-dimensional tile of 1024
with one warp therefore means that 32 CUDA threads cooperate to process 1024
logical elements.

Default logical tiles are:

| Kernel | Default logical tile |
| --- | --- |
| 1-D elementwise or reduction | `1024` |
| 2-D elementwise, stencil, or reduction | `16, 16` |
| f32 matrix multiplication | `16, 16, 32` |
| f64 matrix multiplication | `16, 16, 8` |

Use `FNACC_DEBUG=1` to print the grid, logical tile, and CUDA block selected for
each launch. The generated per-kernel JSON is the authoritative schedule.

## Directive reference

FnACC directives are case-insensitive. The recommended free-form sentinel is
`!$fnacc`. The frontend also accepts `!@fnacc`; fixed-form sentinel handling
depends on the Flang source form in use. The driver primarily scans
conventional FnACC sentinels when deciding whether to run the accelerator
pipeline.

### `parallel`

```fortran
!$fnacc parallel [tile(...)] [pack(...)] [reduction(...)] [no_copyback]
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
persistent allocation keyed by the host data address in the active CUDA
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
- the allocation remains local to its CUDA context.

### `no_copyback`

Use `no_copyback` when the values written by a launch will be consumed by
later device work:

```fortran
!$fnacc parallel tile(16,16) no_copyback
do k = y_min, y_max
  do j = x_min, x_max
    pressure(j,k) = equation_of_state(j,k)
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
Every object must already be present in the active CUDA context; otherwise the
runtime reports an error. `present` neither acquires nested-region ownership
nor performs a transfer.

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

The runtime owns one nonblocking CUDA stream and one completion event per CUDA
context. Work submitted through the same context is ordered in that stream.

Use `!$fnacc wait`:

- before host code that needs completion but not a host transfer;
- before external CUDA work whose stream has no explicit dependency on the
  FnACC stream;
- as a phase boundary before changing device or context ownership; or
- in tests that require completion at a precise source point.

Host updates, final copyouts, reductions that return host scalars, releases,
and host-target launch paths are synchronization points in the current
runtime. `wait` covers only the active FnACC context and its runtime stream; it
does not synchronize unrelated CUDA contexts or caller-created streams.

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

The version 2 launch ABI is variadic. It is not restricted to the older
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

Partial and scratch buffers are cached as grow-only workspaces per CUDA
context. Repeated reductions reuse those allocations.

```sh
FNACC_REDUCTION_STATS=1 ./program
```

prints allocation, growth, reuse, capacity, primary-launch, and stage-launch
counters at process exit. `FNACC_DEBUG=1` prints individual stages.

## Types, ranks, and storage

| Feature | Supported today |
| --- | --- |
| Device expression elements | `real(4)`, `real(8)`, signed `integer(1|2|4|8)` |
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
8. LLVM-IR-to-PTX compilation;
9. device-image and JSON embedding; and
10. host-object generation followed by a relocatable link.

The result of `-c` is a conventional relocatable object containing host code,
the embedded device bundle, metadata, and its registration constructor. No
sidecar PTX or JSON files are required at run time.

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

When it detects FnACC code, the driver adds the equivalent of:

```text
-L$LLVM_BUILD/lib -lFortranFNACCRuntime -lcuda -lstdc++
```

with appropriate runtime search paths. Object and archive inputs are scanned
for FnACC symbols. Use `--fnacc-runtime` when FnACC code is visible only
through `-lNAME` and cannot be detected from a named input file.

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
| `--fnacc-sm N` | NVIDIA target such as `80`, `sm_80`, or `cc80`. |
| `--fnacc-backend NAME` | Preferred backend; default `auto`. |
| `--fnacc-fallback-backend NAME` | Backend used when the preferred backend rejects a kernel; default `triton`. |
| `--fnacc-backend-fallback` | Enable fallback; currently the default. |
| `--fnacc-no-backend-fallback` | Fail instead of using the fallback backend. |
| `--fnacc-num-warps N` | Requested warps per CTA; a power of two and at most 32. |
| `--fnacc-threads-per-warp N` | Subgroup width; CUDA currently requires 32. |
| `--fnacc-num-stages N` | Triton pipeline stages, currently 1 through 16. |
| `--fnacc-f64-matmul-strategy NAME` | `reduce`, `fma`, or `dot`. |
| `--fnacc-cuda-lib-dir DIR` | CUDA Driver API library directory. |
| `--fnacc-workdir DIR` | Parent directory for a unique intermediate tree. |
| `--fnacc-keep` | Keep intermediate files. |
| `--fnacc-verbose` | Print commands before executing them. |
| `--fnacc-dry-run` | Print commands without executing them. |
| `--fnacc-stop-after STAGE` | Stop one FnACC source after an internal stage. |

Compatibility aliases without the `fnacc-` prefix are accepted for schedule,
work-directory, verbosity, and stop controls.

Useful stop stages include `modgen`, `fir`, `fnacc-pipeline`, `ttgir`,
`llvm-mlir`, `llvm-ir`, `ptx`, `embed`, `host-ll`, `host-obj`, `object`,
`objects`, and `link`.

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
| `LLC` | Matching `llc` used to emit PTX. |
| `FLANG`, `FIROPT`, `TCO`, `CLANG` | Optional tool overrides. |
| `FLANG_INTRINSIC_MODULES_PATH` | Override Flang's intrinsic-module directory. |
| `CUDA_LIB_DIR` | CUDA Driver API library directory. |
| `FNACC_SM` | Default NVIDIA target used by the driver. |
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
| `FNACC_ALLOW_EMPTY_KERNELS` | Permit an empty kernel list despite an FnACC `parallel` launch; default false. Data-only sources do not need it. |

The work directory must be writable and contain no whitespace because some
toolchain components and generated command lines require whitespace-free
intermediate paths.

## Runtime configuration

| Variable | Purpose |
| --- | --- |
| `FNACC_CUDA_DEVICE` | CUDA device ordinal. Default `0`; may change between runtime calls. |
| `FNACC_USE_CURRENT_CONTEXT` | Use the caller's current CUDA context instead of retaining a primary context. |
| `FNACC_DEBUG` | Print initialization, bundle, cache, data-region, launch, grid, tile, ABI, and reduction diagnostics. |
| `FNACC_REDUCTION_STATS` | Print aggregate reduction-workspace counters at exit. |
| `FNACC_MATMUL_SHARED_BYTES` | Advanced f32 matmul dynamic-shared-memory override; cannot be below the computed safe minimum. |
| `FNACC_MATMUL_F64_SHARED_BYTES` | Advanced f64 matmul dynamic-shared-memory override; cannot be below the computed safe minimum. |
| `FNACC_PTX_DIR`, `FNACC_PTX`, `FNACC_KERNELS_JSON` | Legacy external-bundle debugging fallbacks. Driver-built objects normally use embedded images and JSON. |

### CUDA context behavior

By default the runtime calls `cuInit(0)`, selects `FNACC_CUDA_DEVICE`, retains
that device's primary context, and restores the caller's previous current
context on return.

With `FNACC_USE_CURRENT_CONTEXT=1`, the caller must make a CUDA context current
before entering FnACC. Runtime state is created for that exact context and the
runtime does not retain or release it. Using the option with no current context
is a fatal error.

CUDA modules, function handles, device allocations, data-region frames,
streams, events, and reduction workspaces are stored per context and are never
reused in another context.

### Thread safety

Public runtime entry points and shared caches are protected by a process-wide
recursive mutex. Calls from multiple host threads are safe with respect to
runtime maps and CUDA resource lifetimes, but host-side runtime entry is
currently serialized. This is correctness-oriented thread safety, not
concurrent multi-stream execution.

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
| CUDA runtime | `flang/lib/Runtime/FNACC/fnacc_runtime.cpp` |
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

### Version 2 launch ABI

The variadic launch interface is built in stages:

1. `__fnacc_begin_launch_v2` creates a pending launch;
2. array descriptors, scalar values, index captures, and reduction results are
   bound with typed `__fnacc_bind_*_v2` calls; and
3. `__fnacc_commit_launch_v2` validates metadata and launches the kernel.

The stable public ABI includes array pointers and layout metadata, scalar and
index values, loop extents and lower bounds, and reduction outputs. Backend
private arguments are excluded. Triton/NVVM currently records its private
pointer count separately in metadata.

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

The only complete backend currently registered is `triton`. Selection supports
`auto`, a named preferred backend, an optional fallback, and detailed fallback
diagnostics. Mixed backends within one emitted device module are not yet
supported.

### JSON metadata

Schema version 1 and backend-contract version 1 include top-level backend and
image fields plus a list of kernel descriptors. Each descriptor records:

- stable bundle-qualified `id` and `name`;
- backend, device-IR kind, and device-image kind;
- image index and file;
- kernel kind and rank;
- logical tile, warp/stage schedule, and CUDA CTA size;
- launch ABI version;
- array, scalar, output, and reduction-result counts;
- parameter roles, slots, names, types, array dimensions, and layout fields;
- loop lower-bound and extent roles;
- pack bindings and `copy_back_writes`;
- backend-private pointer count; and
- reduction operator and synthetic-stage identity where applicable.

Legacy PTX and Triton-private field aliases remain while older tools are being
retired.

### Runtime images

The typed registration ABI accepts PTX and cubin images. The normal Triton
driver path emits PTX. A future direct-PTX, CUDA Tile IR, or other backend may
reuse the kernel plan, public ABI, metadata, embedding, and runtime dispatch
layers, but its driver must still produce a runtime-supported image.

Adding a compiler backend enum alone is insufficient. Driver dispatch,
manifest/embed logic, runtime loading, contract validation, and tests must be
updated together.

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
and negative diagnostics.

Prefer `CHECK-LABEL`, `CHECK-NEXT`, bounded `CHECK`, and `CHECK-DAG` patterns to
fragile `CHECK-SAME` assertions when operations are intentionally printed on
separate lines. Tests should verify semantics and types rather than local SSA
names.

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
fnacc-flang --fnacc-keep --fnacc-verbose -c kernel.f90
FNACC_DEBUG=1 FNACC_CUDA_DEVICE=0 ./program
```

For memory checking:

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
4. Add runtime binding when the existing variadic launcher cannot represent
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

### Change a runtime ABI

Update these together:

- runtime-call creation in `FNACCLowerToRuntime.cpp`;
- exported runtime function signatures;
- JSON parameter roles and types;
- driver image/ABI validation;
- compatibility aliases or a schema/ABI version; and
- MLIR lowering and executable integration tests.

CUDA-owned objects must remain in state keyed by the active `CUcontext`; never
infer ownership from a process-global device-pointer cache.

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
owning data region, belongs to another CUDA context, or was used only through
a host-temporary launch. Establish storage with `enter data`, `create`,
`update device`, or `pack(...:device)` before asserting presence or updating
the host.

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
grid, tile, CUDA block size, and image metadata.

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
- Triton is the only complete code-generation backend. The normal path emits
  PTX; cubin is accepted by the typed runtime image ABI but is not the normal
  Triton driver output.
- Mixed backends in one device module are unsupported.
- The runtime owns one stream per CUDA context and serializes public entry
  through a process-wide mutex.
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

