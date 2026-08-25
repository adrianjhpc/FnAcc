# FnACC

FnACC is a programming approach (think OpenACC or OpenMP) to enable porting Fortran programs to GPUs (or other accelerators). It is designed to leverage [Triton](https://triton-lang.org/main/index.html) to port to the accelerator(s), although it is being extended to other backends as well.


FnACC is implemented using LLVM's Flang. *Note: The FnACC compiler source code isn't in this repository. It currently lives in the FnACC branch in a fork of LLVM here:(https://github.com/adrianjhpc/llvm-project/tree/FnACC)[https://github.com/adrianjhpc/llvm-project/tree/FnACC]. FnACC currenlty recognises a deliberately constrained set of Fortran loop kernels, builds a backend-neutral kernel plan, emits device code, and replaces the host region with calls to a CUDA Driver API runtime.

The production path currently implemented end to end is:

```text
Fortran + !$fnacc
  -> Flang parse tree and FIR
  -> fnacc.launch and FNACC data operations
  -> kernel recognition and backend-neutral planning
  -> Triton TTIR -> TTGIR -> LLVM MLIR -> LLVM IR -> PTX
  -> embedded device-image/JSON bundle
  -> host object + FNACC CUDA runtime
```

FnACC is a prototype. Its syntax and runtime behaviour are intentionally small and may change. Treat the source, generated JSON schema, and regression tests as the authoritative contract.

## Contents

- [Current capabilities](#current-capabilities)
- [Quick start](#quick-start)
- [Programming model](#programming-model)
- [Directive reference](#directive-reference)
- [Supported kernels and expressions](#supported-kernels-and-expressions)
- [Data types, ranks, and storage](#data-types-ranks-and-storage)
- [Scalar capture and private temporaries](#scalar-capture-and-private-temporaries)
- [Reductions](#reductions)
- [Data movement and lifetime](#data-movement-and-lifetime)
- [Synchronization and `wait`](#synchronization-and-wait)
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

- a `parallel` directive for canonical one- and two-dimensional Fortran loops;
- elementwise expressions over one to three input arrays and one output array;
- up to three read-only scalar captures;
- `real(4)`, `real(8)`, and signed `integer(1|2|4|8)` elementwise kernels;
- specialised single- and double-precision two-dimensional matrix
  multiplication;
- sum, dot-product, product, minimum, and maximum reductions;
- floating-point and integer reductions;
- hierarchical, multi-kernel device reductions with reusable workspaces;
- one- and multi-warp reduction lowering;
- explicit persistent data management and `pack` placement;
- explicit synchronization with `!$fnacc wait`;
- explicit-shape, assumed-shape, pointer/heap-backed, and allocatable arrays
  when their storage is contiguous;
- per-device and per-CUDA-context runtime state;
- a thread-safe runtime cache interface;
- embedded PTX or cubin bundle registration through a typed device-image ABI;
- a backend-neutral kernel plan and artifact contract, with Triton as the only
  complete code-generation backend at present.

## Quick start

### Toolchain environment

The compiler driver needs an FnACC-enabled LLVM/Flang build and the Triton
lowering tools:

```sh
export LLVM_BUILD=/path/to/llvm-project/build
export TRITON_OPT=/path/to/triton-opt
export MLIR_TRANSLATE=/path/to/triton-llvm/build/bin/mlir-translate
export LLC=/path/to/triton-llvm/build/bin/llc

# Optional when libcuda is not in the default linker search path.
export CUDA_LIB_DIR=/usr/local/cuda/lib64
```

Build the FnACC runtime in the LLVM tree with `FLANG_FNACC_RUNTIME=ON`. The
compatibility spelling `FLANG_FNACC` is also accepted by the current CMake
configuration.

Typical rebuild targets are:

```sh
cmake --build "$LLVM_BUILD" --target fir-opt flang FortranFNACCRuntime
```

### A complete vector-add example

```fortran
module vector_kernels
  implicit none
contains
  subroutine vector_add(n, a, b, c)
    integer, intent(in) :: n
    real, intent(in) :: a(n), b(n)
    real, intent(out) :: c(n)
    integer :: i

    !$fnacc parallel tile(256) pack(a:device, b:device, c:device)
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
  call vector_add(n, a, b, c)
  !$fnacc wait
  !$fnacc exit data copyout(c) delete(a, b, c)

  if (any(c /= 3.0)) error stop "validation failed"
end program
```

Compile, link, and run it with:

```sh
/path/to/FnAcc/bin/fnacc-flang example.f90 -O3 -o example
FNACC_CUDA_DEVICE=0 ./example
```

For separate compilation, keep all accelerator procedures in one FnACC-bearing
translation unit under the current one-bundle restriction:

```sh
fnacc-flang -O3 -c kernels.f90 -o kernels.o
fnacc-flang -O3 main.f90 kernels.o -o example
```

## Programming model

`!$fnacc parallel` applies to the immediately following `do` construct. There
is no matching end directive.

```fortran
!$fnacc parallel tile(128)
do i = 1, n
  c(i) = a(i) + b(i)
end do
```

The compiler does not lower an arbitrary region of Fortran. It recognises the
loop, array accesses, scalar values, expression tree, and side effects as one
supported kernel. Compilation fails if it cannot account for the whole launch
region. This all-or-nothing rule is important: unsupported statements are not
silently discarded or left to execute on the host.

### Canonical loop requirements

The current recognisers require:

- lower bound `1`;
- constant step `1`;
- a recoverable upper extent;
- exactly one top-level loop for a rank-one kernel;
- exactly one outer and one inner loop for a rank-two kernel;
- regular contiguous array addressing based on the recognised induction
  variables.

For two-dimensional kernels, the outer loop is normally the second Fortran
dimension and the inner loop the first:

```fortran
!$fnacc parallel tile(16, 16)
do j = 1, m
  do i = 1, n
    c(i, j) = alpha * a(i, j) + beta * b(i, j)
  end do
end do
```

### Tile size and CUDA block size are different

`tile(...)` describes the logical number of elements handled by one device
program in each dimension. It is not the CUDA thread-block size.

The CUDA block size comes from the selected schedule:

```text
cuda_threads_per_cta = num_warps * threads_per_warp
```

CUDA currently requires `threads_per_warp=32`. A one-dimensional tile of 1024
with one warp therefore means 32 CUDA threads cooperate to process 1024 logical
elements. Use `FNACC_DEBUG=1` to see both the grid, logical tile, and CUDA block
for each launch.

Default logical tiles are currently:

| Kernel | Default tile |
| --- | --- |
| 1-D elementwise or reduction | `1024` |
| 2-D elementwise | `16, 16` |
| f32 matrix multiplication | `16, 16, 32` |
| f64 matrix multiplication | `16, 16, 8` |

Explicit tile sizes are recommended in tests and performance-sensitive code.
The planner may choose a different effective warp count for some simple
elementwise kernels; generated per-kernel JSON is the authoritative schedule.

# FnACC Programmer Guide

## Directive reference

The standard and recommended source sentinel is `!$fnacc`. The parser also
accepts `!@fnacc` and `!dir$ fnacc` forms. Use `!$fnacc` in portable project
sources because the compiler driver's source auto-detection is based primarily
on conventional FnACC sentinels. The driver also scans `c$fnacc` and
`*$fnacc` when deciding whether to invoke the accelerator pipeline; frontend
acceptance of those spellings still depends on fixed-form handling in the
selected Flang build.

### Parallel directive

```fortran
!$fnacc parallel [tile(...)] [pack(...)] [reduction(...)]
do ...
  ...
end do
```

| Clause | Purpose |
| --- | --- |
| `tile(x[, y[, z]])` | Compile-time logical tile shape. `z` is the reduction/K tile for matrix multiplication. |
| `pack(name:host, ...)` | Use a temporary device buffer and host-visible launch behaviour for the named kernel arguments. |
| `pack(name:device, ...)` | Use or create a persistent cached device allocation for the named kernel arguments. |
| `reduction(+:s)` | Additive reduction into scalar `s`. |
| `reduction(*:s)` | Multiplicative reduction into scalar `s`. |
| `reduction(min:s)` | Minimum reduction into scalar `s`. |
| `reduction(max:s)` | Maximum reduction into scalar `s`. |

Although the grammar can represent a list of reduction items, the current
kernel/runtime path accepts one reduction scalar and one operator per launch.
`pack` and data-directive lists currently contain bare variable names, not
array sections. Placement metadata is meaningful for array storage; scalar
expression captures are passed by value.

### Data and synchronization directives

| Directive | Current behaviour |
| --- | --- |
| `!$fnacc enter data copyin(a, b)` | Allocate persistent device storage and copy the current host values to it. |
| `!$fnacc enter data create(c)` | Allocate persistent device storage without copying host data. |
| `!$fnacc update device(a, b)` | Create/resize persistent storage as needed, then copy host values to the device. |
| `!$fnacc update host(c)` | Copy a present persistent device allocation to the host. Fails if the object is not present. |
| `!$fnacc exit data copyout(c)` | Copy a present allocation to the host. It does not by itself delete the allocation. |
| `!$fnacc exit data delete(a, b, c)` | Release the named persistent allocations without a copy. |
| `!$fnacc release(a, b)` | Release the named persistent allocations; equivalent to the runtime lifetime action of `delete`. |
| `!$fnacc release all` | Release all persistent allocations in the active CUDA context. `release_all` is an accepted alias. |
| `!$fnacc wait` | Wait for prior work on the FnACC stream belonging to the active CUDA context. No data is transferred. |

Clauses on `enter data` and `exit data` are processed in source order. A common
exit sequence is therefore:

```fortran
!$fnacc exit data copyout(c) delete(a, b, c)
```

## Supported kernels and expressions

### Elementwise kernels

Rank-one and rank-two elementwise assignments are supported with:

- one to three read arrays;
- one write array;
- zero to three read-only scalar captures;
- one expression assigned to each output element;
- all array elements and scalar expression values using one supported element
  type.

Examples include:

```fortran
c(i) = a(i) + b(i)
c(i) = a(i) + b(i) + d(i)
c(i) = alpha * a(i) + beta * b(i)
c(i, j) = max(lower, min(a(i, j), upper))
c(i) = merge(a(i), b(i), a(i) >= b(i))
```

### Floating-point expression operations

| Category | Supported operations |
| --- | --- |
| Arithmetic | `+`, `-`, `*`, `/`, unary negation |
| Unary intrinsics | `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh` |
| Min/max | `min`, `max`, plus recognised numeric min/max FIR forms |
| Comparisons | `<`, `<=`, `>`, `>=`, `==`, `/=` using ordered floating-point comparisons |
| Selection | `merge(true_value, false_value, mask)` when lowered to a supported select tree |

### Integer expression operations

Signed `integer(1)`, `integer(2)`, `integer(4)`, and `integer(8)` expressions
support:

- addition, subtraction, multiplication, and signed division;
- `abs`;
- signed `min` and `max`;
- signed relational comparisons and equality/inequality;
- `merge`/selection over supported integer expressions.

Mixed element types, implicit widening inside the device expression, unsigned
integer semantics, complex values, logical arrays, character values, derived
types, and arbitrary calls are not currently part of the kernel subset.

### Matrix multiplication

The recogniser accepts the canonical three-loop form:

```fortran
!$fnacc parallel tile(16, 16, 32) pack(a:device, b:device, c:device)
do j = 1, n
  do i = 1, m
    acc = 0.0
    do p = 1, k
      acc = acc + a(i, p) * b(p, j)
    end do
    c(i, j) = acc
  end do
end do
```

`real(4)` and `real(8)` matrices are supported. For f64, select the code
generation strategy with:

```sh
fnacc-flang --fnacc-f64-matmul-strategy reduce ...
fnacc-flang --fnacc-f64-matmul-strategy fma ...
fnacc-flang --fnacc-f64-matmul-strategy dot ...
```

Strategy performance and toolchain compatibility are GPU- and Triton-version
dependent. Validate numerical behaviour and benchmark all three on the target
system.

## Data types, ranks, and storage

| Feature | Supported today |
| --- | --- |
| Element types | `real(4)`, `real(8)`, signed `integer(1|2|4|8)` |
| Elementwise kernel rank | 1 or 2 |
| Matrix multiplication rank | 2 |
| Reduction rank | 1 |
| Descriptor data operations | Rank 1 through 3 |
| Storage | Contiguous explicit-shape, assumed-shape, pointer/heap-backed, and allocatable arrays |
| Runtime extent ABI | Signed 32-bit extents |

The data-operation lowering has two paths:

1. FIR descriptors are lowered to a descriptor ABI containing the data
   pointer, element size, rank, extents, and byte strides.
2. Sized scalar and explicit-shape references are lowered to a raw pointer and
   byte count.

The runtime validates descriptor contiguity. Noncontiguous array sections,
strided sections, and ranks beyond the stated limits are rejected or cannot be
sized safely. Release a cached allocatable before deallocating or reallocating
it so that a stale host-address key is not retained.

## Scalar capture and private temporaries

FnACC does not currently have a source-level `private` clause. Scalar
references are classified automatically.

### Read-only capture

A scalar that is only read by the kernel is loaded on the host and passed by
value. This is similar to `firstprivate` behaviour:

```fortran
! alpha and beta are scalar kernel arguments.
c(i) = alpha * a(i) + beta * b(i)
```

### Iteration-private temporary

A mutable scalar reference is promoted to device SSA and treated as private to
one logical iteration only when the compiler proves all of the following:

- it has exactly one defining store in the same innermost loop block;
- that store dominates every load in the iteration;
- it is not read before being written;
- it is not loop-carried or conditionally defined;
- it is not used by host code after the launch.

For example:

```fortran
! tmp is promoted and is not a shared pointer kernel argument.
!$fnacc parallel tile(128)
do i = 1, n
  tmp = alpha * a(i)
  c(i) = tmp + b(i)
end do
```

### Unsupported mutable reference

Compilation fails with a diagnostic such as
`mutable scalar reference is neither iteration-private nor a reduction` when a
scalar has multiple stores, a read-before-write dependency, cross-block or
cross-loop use, conditional definition, or a host-observed value after the
launch. Express intentional cross-iteration accumulation with a supported
`reduction` clause instead.

## Reductions

FnACC recognises these one-dimensional patterns:

```fortran
sum = 0.0
!$fnacc parallel tile(256) reduction(+:sum)
do i = 1, n
  sum = sum + a(i)
end do

dot = 0.0
!$fnacc parallel tile(256) reduction(+:dot)
do i = 1, n
  dot = dot + a(i) * b(i)
end do

product = 1.0
!$fnacc parallel tile(256) reduction(*:product)
do i = 1, n
  product = product * a(i)
end do

smallest = huge(smallest)
!$fnacc parallel tile(256) reduction(min:smallest)
do i = 1, n
  smallest = min(smallest, a(i))
end do

largest = -huge(largest)
!$fnacc parallel tile(256) reduction(max:largest)
do i = 1, n
  largest = max(largest, a(i))
end do
```

The initial host scalar participates in the result. For an empty extent, the
result remains the initial value. Choose an initialization appropriate to the
operator and element type.

### Hierarchical implementation

The primary kernel writes one partial per logical device program. A synthetic
`reduction_stage1d` kernel then reduces those partials recursively until one
value remains. The final device result is combined with the initial scalar and
returned to the host.

Partial and scratch buffers are cached as grow-only workspaces per CUDA
context. Repeated reductions therefore reuse allocations. Set:

```sh
FNACC_REDUCTION_STATS=1 ./program
```

to print allocation, growth, reuse, capacity, primary-launch, and stage-launch
counters at process exit. `FNACC_DEBUG=1` prints each hierarchical stage.

Multi-warp reduction lowering is automatic. The driver also handles residual
Triton `ttg.warp_id` lowering where required; there is no user compatibility
flag in the current driver.

## Data movement and lifetime

### Default or `pack(...:host)` behaviour

When no persistent allocation is present and no explicit device target is
selected, launch arguments use temporary device buffers:

- read arrays are copied host to device;
- the kernel runs;
- host-target outputs are copied device to host;
- temporary buffers are freed.

This mode is simple but transfer and allocation overhead dominates small
kernels and repeated launches.

### `pack(...:device)` behaviour

Device-target arguments use a persistent allocation cached by host data
address in the active CUDA context:

- a read allocation is created and copied on the first cache miss;
- an output allocation is created without a copy on the first cache miss;
- later launches reuse the allocation;
- device-target outputs are not copied back automatically;
- host writes made after the first copy are not visible until `update device`;
- device results are not visible on the host until `update host` or `copyout`;
- `delete`, `release`, or `release all` ends the cached lifetime.

An explicit data directive also makes an allocation present. Subsequent
launches use present cached storage even when they omit an explicit
`pack(...:device)` item. This present-if-cached rule lets data regions control
placement centrally.

### Recommended persistent-data pattern

```fortran
!$fnacc enter data copyin(a, b) create(c)

do step = 1, timesteps
  call kernel_a(n, a, b, c)
  call kernel_b(n, c, a)
end do

!$fnacc wait
!$fnacc exit data copyout(a, c) delete(a, b, c)
```

If host code changes `a` during the data lifetime, add:

```fortran
!$fnacc update device(a)
```

before the next device consumer.

## Synchronization and `wait`

The runtime owns one nonblocking CUDA stream and one completion event per CUDA
context. Work submitted through the same FnACC context is ordered in that
stream.

`!$fnacc wait` waits for all prior FnACC work in the active context's runtime
stream. It does not copy data and it has no queue or stream operand.

Use `wait`:

- before host code that relies on completion but no host update/copyout is needed;
- before handing FnACC-managed device data to external CUDA work whose stream
  has no explicit dependency on the FnACC stream;
- as an explicit phase boundary before changing context/device ownership;
- in tests that need deterministic completion at a particular source point.

It is not needed between ordinary consecutive FnACC launches on the same
context because stream order preserves their dependency. Reductions that
return a host scalar, host-target launch paths, host updates/copyouts, and
release operations are synchronization points in the current runtime.

Most current launch wrappers already wait on their completion event before
returning, so `wait` is often redundant today. Keeping the directive at true
host/device phase boundaries documents the dependency and remains correct as
more device-target paths become asynchronous.

`wait` only covers the active FnACC context and its single runtime stream. It
does not synchronise every CUDA device, arbitrary caller streams, or unrelated
CUDA contexts.

## Compilation and linking

### Driver pipeline

For an FnACC source, `fnacc-flang` performs:

1. a syntax-only Flang invocation to generate module files;
2. FIR emission;
3. the `fnacc-pipeline` pass pipeline, producing host FIR, device IR, and JSON;
4. per-kernel device-IR splitting;
5. TTIR to TritonGPU IR lowering for the Triton backend;
6. TritonGPU IR to LLVM MLIR lowering;
7. LLVM MLIR to LLVM IR translation;
8. LLVM IR to PTX compilation;
9. device-image/JSON embedding and host-object generation;
10. a relocatable link that combines the host and embedded objects.

The result of `-c` is a conventional relocatable object containing host code,
the embedded device bundle, JSON metadata, and its registration constructor.
No sidecar PTX or JSON files are required at run time.

### Ordinary Fortran sources

Sources without a recognised FnACC sentinel are delegated to Flang. `-E`,
`-S`, and `-fsyntax-only` are also delegated and do not run accelerator code
generation. Use `--fnacc-force` or `--fnacc-disable` to override source
auto-detection.

### Linking

The driver automatically adds:

```text
-L$LLVM_BUILD/lib -lFortranFNACCRuntime -lcuda -lstdc++
```

and suitable runtime search paths when it detects FnACC code. Use
`--fnacc-runtime` when the only FnACC-bearing input is hidden behind `-lNAME`.

The runtime currently supports one embedded FnACC bundle per executable. The
driver detects multiple bundle markers and rejects the link. Put all device
kernels needed by an executable in one FnACC-bearing source/module, or combine
them before producing the embedded bundle. Any number of ordinary host objects
may be linked with that single FnACC object.

## Compiler-driver reference

### Traditional options

The driver accepts normal compile/link options including `-c`, `-o`, `-I`,
`-J`, `-L`, `-l`, `-D`, `-U`, `-O`, `-g`, `-f...`, `-m...`, `-Wl,...`, and
`--`. Flags are routed to the relevant Flang, host-codegen, or final-link step.

### FnACC options

| Option | Meaning |
| --- | --- |
| `--fnacc-force` | Use the FnACC pipeline for every Fortran source. |
| `--fnacc-disable` | Delegate every source to Flang. |
| `--fnacc-runtime` | Force the runtime libraries into the final link. |
| `--fnacc-no-runtime` | Do not auto-add the runtime. |
| `--fnacc-sm N` | NVIDIA target, for example `80`, `sm_80`, or `cc80`. |
| `--fnacc-backend NAME` | Preferred backend; current default is `auto`. |
| `--fnacc-fallback-backend NAME` | Backend used if the preferred backend rejects a kernel; default `triton`. |
| `--fnacc-backend-fallback` | Enable fallback; currently the default. |
| `--fnacc-no-backend-fallback` | Fail instead of falling back. |
| `--fnacc-num-warps N` | Requested warps per CTA; power of two, at most 32. |
| `--fnacc-threads-per-warp N` | Subgroup width; CUDA currently requires 32. |
| `--fnacc-num-stages N` | Triton pipeline stages, currently 1 through 16. |
| `--fnacc-f64-matmul-strategy NAME` | `reduce`, `fma`, or `dot`. |
| `--fnacc-cuda-lib-dir DIR` | CUDA Driver API library directory. |
| `--fnacc-workdir DIR` | Parent directory for a unique intermediate tree. |
| `--fnacc-keep` | Keep intermediate files. |
| `--fnacc-verbose` | Print executed commands. |
| `--fnacc-dry-run` | Print commands without running them. |
| `--fnacc-stop-after STAGE` | Stop one FnACC source after an internal stage. |

Compatibility aliases without the `fnacc-` prefix are accepted for the
schedule, work-directory, verbosity, and stop controls.

Useful stop stages include `modgen`, `fir`, `fnacc-pipeline`, `ttgir`,
`llvm-mlir`, `llvm-ir`, `ptx`, `embed`, `host-ll`, `host-obj`, `object`,
`objects`, and `link`.

For example:

```sh
fnacc-flang --fnacc-verbose --fnacc-keep \
  --fnacc-stop-after fnacc-pipeline -c kernels.f90
```

prints the exact `fir-opt` command and preserves FIR, TTIR, JSON, and the
toolchain manifest.

### Driver environment variables

| Variable | Purpose |
| --- | --- |
| `LLVM_BUILD` | FnACC-enabled LLVM/Flang build directory. |
| `TRITON_OPT` | `triton-opt` executable. Required by the Triton backend. |
| `MLIR_TRANSLATE` | Matching `mlir-translate`. |
| `LLC` | Matching `llc` used to emit PTX. |
| `FLANG`, `FIROPT`, `TCO`, `CLANG` | Optional tool overrides. |
| `FLANG_INTRINSIC_MODULES_PATH` | Override Flang's intrinsic-module directory. |
| `CUDA_LIB_DIR` | CUDA Driver API library directory. |
| `FNACC_SM` | Default SM target; currently 80. |
| `FNACC_BACKEND` | Preferred backend; default `auto`. |
| `FNACC_FALLBACK_BACKEND` | Fallback backend; default `triton`. |
| `FNACC_ALLOW_BACKEND_FALLBACK` | Boolean backend-fallback control. |
| `FNACC_NUM_WARPS` | Default requested warp count; currently 1. |
| `FNACC_THREADS_PER_WARP` | Default subgroup width; currently 32. |
| `FNACC_NUM_STAGES` | Default pipeline stages; currently 3. |
| `FNACC_F64_MATMUL_STRATEGY` | Default f64 matmul strategy. |
| `FNACC_WORKDIR` | Default intermediate-directory parent. |
| `FNACC_TTIR_TO_TTGIR_PASSES` | Advanced override for the TTIR-to-TTGIR pass pipeline. |
| `FNACC_TTGIR_TO_LLVM_PASSES` | Advanced override for the TTGIR-to-LLVM-MLIR pass pipeline. |
| `FNACC_ALLOW_EMPTY_KERNELS` | Permit empty generated kernel JSON for diagnostics; default false. |

The work directory must be writable and contain no whitespace because several
toolchain components and generated command lines require whitespace-free
intermediate paths.

## Runtime configuration

| Variable | Purpose |
| --- | --- |
| `FNACC_CUDA_DEVICE` | CUDA device ordinal. Default `0`; may be changed between runtime calls. |
| `FNACC_USE_CURRENT_CONTEXT` | Use the caller's current CUDA context instead of retaining a primary context. |
| `FNACC_DEBUG` | Print runtime initialization, cache, launch, grid, tile, ABI, and reduction-stage diagnostics. |
| `FNACC_REDUCTION_STATS` | Print aggregate reduction workspace counters at exit. |
| `FNACC_MATMUL_SHARED_BYTES` | Advanced f32 matmul dynamic-shared-memory override; may not be below the computed safe minimum. |
| `FNACC_MATMUL_F64_SHARED_BYTES` | Advanced f64 matmul dynamic-shared-memory override; may not be below the computed safe minimum. |
| `FNACC_PTX_DIR`, `FNACC_PTX`, `FNACC_KERNELS_JSON` | Legacy external-bundle debugging fallbacks. Normal driver-built objects use embedded images and JSON. |

### CUDA context behaviour

By default the runtime calls `cuInit(0)`, selects `FNACC_CUDA_DEVICE`, retains
that device's primary context, and restores the caller's prior current context
on return.

With `FNACC_USE_CURRENT_CONTEXT=1`, the caller must make a CUDA context current
before entering FnACC. The runtime creates state for that exact context and
does not retain or release it. Passing this option with no current context is a
fatal error.

CUDA modules, function handles, device allocations, streams, events, and
reduction workspaces are stored per context. They are never reused in another
context, even if two contexts select the same physical device.

### Thread safety

Public runtime entry points and shared caches are protected by a process-wide
recursive mutex. Calls from multiple host threads are safe with respect to the
runtime's internal maps and CUDA resource lifetimes, but the lock currently
serialises FnACC host-side runtime entry. This is correctness-oriented thread
safety, not concurrent multi-stream execution.

## Compiler and runtime architecture

### Frontend and FIR

The main implementation areas are:

| Area | Principal files |
| --- | --- |
| Parse tree and syntax | `flang/include/flang/Parser/parse-tree-fnacc.h`, `flang/lib/Parser/fnacc-parsers.cpp` |
| Unparsing and semantics | `flang/lib/Parser/unparse.cpp`, `flang/lib/Semantics/resolve-names.cpp` |
| PFT and FIR generation | `flang/include/flang/Lower/PFTBuilder.h`, `flang/lib/Lower/PFTBuilder.cpp`, `flang/lib/Lower/Bridge.cpp` |
| FNACC dialect | `flang/include/flang/Optimizer/Dialect/FNACC/FNACCOps.td`, `FNACCDialect.td`, `flang/lib/Optimizer/Dialect/FNACC/FNACCDialect.cpp` |
| Recognition and planning | `FNACCKernelAnalysis.h/.cpp`, `FNACCKernelPlan.h` |
| Triton backend | `flang/lib/Optimizer/Dialect/FNACC/FNACCLowerToTriton.cpp` |
| Host runtime lowering | `flang/lib/Optimizer/Dialect/FNACC/FNACCLowerToRuntime.cpp` |
| Pass pipeline | `FNACCPasses.td`, `FNACCPipelines.cpp` |
| CUDA runtime | `flang/lib/Runtime/FNACC/fnacc_runtime.cpp` |
| Traditional driver | `FnAcc/bin/fnacc-flang` |

The FIR dialect includes:

- `fnacc.launch` with constant tile sizes, packed values, and target metadata;
- `fnacc.terminator` for its single-block region;
- `fnacc.update_host`, `fnacc.update_device`;
- `fnacc.copyin`, `fnacc.create`, `fnacc.copyout`, `fnacc.delete`;
- `fnacc.release`, `fnacc.release_all`, and `fnacc.wait`.

Handwritten MLIR tests must terminate `fnacc.launch` with
`fnacc.terminator`; `fir.end` is not the FNACC region terminator.

### Pass pipeline

The registered `fnacc-pipeline` performs:

1. `fnacc-assign-kernel-ids` for stable IDs and symbols;
2. kernel recognition, planning, backend selection, device IR, and JSON
   emission;
3. `fnacc-lower-to-runtime` for host launch/data calls;
4. optional `fnacc-emit-fortran-aliases` for external Fortran ABI aliases.

`fnacc-outline-kernels` also exists for development experiments but is not a
normal stage of the default compilation route.

### Kernel recognition and consumed operations

Recognition builds an `ElementwiseKernel` containing loop and extent sources,
read/write arrays, scalar captures, reduction metadata, expression tree, and a
list of consumed FIR operations. Validation requires every operation with
observable effects in the launch region to be either consumed by that kernel
or accepted as structural loop/terminator machinery.

When adding a new pattern, update the consumed-operation accounting together
with recognition. Otherwise lowering may reject a valid-looking pattern—or,
more seriously, a future change could erase an operation whose effect was not
represented in device code.

### Host ABI

The backend-neutral ABI orders parameters by role:

- read device pointers;
- write or partials device pointer;
- scalar values;
- `extent_x`, optional `extent_y`, and matrix `extent_k`.

Backend-private arguments are explicitly excluded from this stable ABI. The
Triton/NVVM path currently appends two private pointer parameters, represented
in metadata as `private_pointer_args=2` with the legacy
`triton_hidden_ptr_args` alias retained for compatibility.

## Backend artifact contract

Kernel recognition and scheduling are backend-neutral. `FNACCKernelPlan`
contains stable identity, the recognised expression/kernel, tile and subgroup
schedule, public ABI, pack bindings, and an optional synthetic reduction-stage
plan.

`FNACCCodegenBackend` provides:

- a backend name;
- emitted device-IR kind;
- final runtime image kind;
- per-plan support queries and rejection reasons;
- module framing and kernel emission;
- the count of backend-private pointer arguments.

The current registered backend is `triton`. The selection interface supports
`auto`, a named preferred backend, a named fallback, and fallback diagnostics.
Mixed backends within one emitted module are not supported yet.

### JSON metadata

Schema version 1 and backend-contract version 1 include top-level fields such
as:

- requested, fallback, and selected backend;
- whether fallback was allowed or used;
- emitted device-IR kind;
- runtime device-image kind;
- per-kernel descriptors.

Each kernel descriptor includes:

- stable `id` and `name`;
- `backend`, `device_ir_kind`, and `device_image_kind`;
- `image_index` and `image_file`;
- kernel `kind`, rank, logical tile, warp/stage schedule, and CUDA CTA size;
- stable parameter roles, slots, names, and types;
- pack bindings;
- private pointer count;
- reduction operator and synthetic stage ID when applicable.

Legacy `ptx_index`, `ptx_file`, and `triton_hidden_ptr_args` aliases remain in
the metadata while older drivers/runtimes are being retired.

### Runtime images

The typed embedded-bundle registration ABI tags each image. The runtime
currently accepts PTX and cubin images. The ordinary driver path emits PTX from
the Triton backend. A future CUDA Tile IR or direct-PTX backend can reuse the
kernel plan, public ABI, metadata, embedding, and runtime dispatch layers, but
it must still provide a driver pipeline that converts its device IR into a
runtime-supported PTX or cubin image.

Adding an enum value or a compiler backend alone is not sufficient: update the
driver dispatch, manifest/embed logic, runtime image loader, contract
validation, and tests together.

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
  /path/to/llvm-project/flang/test/FNACC
```

The suite includes parser/unparser tests, FIR lowering, runtime-call lowering,
TTIR and JSON checks, expressions, integers, rank-one/rank-two arrays,
assumed-shape and allocatable descriptors, matmul variants, reductions,
multi-warp lowering, data directives, `wait`, private scalar classification,
and negative diagnostics.

Use `CHECK-LABEL`, `CHECK-NEXT`, or bounded `CHECK` patterns instead of fragile
`CHECK-SAME` assertions when matching operations that are intentionally on
separate TTIR lines. Tests should verify semantics and types rather than local
SSA names.

### Reduction validation executable

The standalone reduction suite is compiled by CMake and the runner script only
executes the prebuilt binary:

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

The test covers f32/f64 sum, dot, product, minimum, and maximum reductions;
small and boundary sizes; non-power-of-two sizes; hierarchical stages; and
workspace allocation, growth, and reuse. Configure with
`-DFNACC_NUM_WARPS=4` to exercise multi-warp reductions.

### Repeated BabelStream measurements

`tools/fnacc-babelstream-stats.py` runs warm-ups and repeated measured trials,
reports minimum/median/mean/maximum bandwidth, spread, coefficient of
variation, median time, and robust outliers, and can preserve machine-readable
JSON:

```sh
tools/fnacc-babelstream-stats.py \
  --warmups 1 --runs 9 \
  --arraysize 33554432 --numtimes 100 \
  --json fnacc-babelstream.json \
  ./BabelStream.fnacc.FnACCArray
```

Use the same device visibility, device ordinal, clocks, array size, iteration
count, and warm-up policy for every implementation being compared. Treat
validation errors as a failed benchmark, regardless of reported bandwidth.

### Runtime debugging

Keep compiler intermediates and enable runtime traces:

```sh
fnacc-flang --fnacc-keep --fnacc-verbose -c kernel.f90
FNACC_DEBUG=1 FNACC_CUDA_DEVICE=0 ./program
```

For memory checking:

```sh
CUDA_LAUNCH_BLOCKING=1 compute-sanitizer \
  --tool memcheck --error-exitcode 99 ./program
```

When benchmarking asynchronous or cached paths, validate results separately
from timed launches and ensure any required host update is outside the timed
region.

## Extending FnACC

### Add or change directive syntax

1. Add parse-tree nodes in `parse-tree-fnacc.h`.
2. Add parsers in `fnacc-parsers.cpp` and executable-construct routing if
   needed.
3. Resolve every contained `Name` explicitly in `resolve-names.cpp`.
4. Add unparse support in `unparse.cpp`.
5. Add PFT/Bridge lowering to an existing or new FNACC FIR operation.
6. Define/verify the operation in `FNACCOps.td` and `FNACCDialect.cpp`.
7. Lower it to the runtime or consume it in kernel planning.
8. Add parser, unparser, FIR, runtime-lowering, and negative tests.

### Add an expression operation

1. Extend `ElementwiseExprKind`.
2. Recognise the exact FIR operation or Flang lowering idiom in
   `FNACCKernelAnalysis.cpp`.
3. Enforce element/predicate result kinds and type restrictions.
4. Mark all recognised operations as consumed.
5. Emit the expression in each applicable backend.
6. Add 1-D and 2-D tests, type tests, edge cases, and an unsupported-type test.

Be careful with intrinsics: Flang may lower one Fortran intrinsic differently
for different kinds or optimization levels. Integer `abs`, min/max, and
`merge` are examples where matching only one apparent FIR spelling is too
fragile.

### Add a kernel pattern

1. Define a kernel kind and recognition result.
2. Prove loop bounds, extents, access ranks, types, alias/side-effect rules, and
   scalar classification.
3. Construct the backend-neutral schedule and public ABI.
4. Add runtime lowering and a stable launcher ABI if the existing generic
   launcher cannot represent it.
5. Implement backend emission and `querySupport` checks.
6. Emit and validate JSON metadata.
7. Add end-to-end numerical tests as well as FIR/TTIR/JSON tests.

Fail closed. A clear compile-time diagnostic is preferable to emitting a
kernel whose indexing, mutation, or ABI has not been proven safe.

### Add a device backend

1. Implement `FNACCCodegenBackend` against `FNACCKernelPlan`.
2. Give unsupported plans precise `querySupport` diagnostics.
3. Register backend selection and fallback behaviour.
4. Emit schema-v1/backend-contract-v1 metadata.
5. Add a driver dispatch for the emitted device-IR and image kinds.
6. Extend typed bundle generation if a new runtime image type is required.
7. Extend runtime validation/module loading.
8. Test preferred, automatic, fallback, disabled-fallback, unsupported, and
   mixed-backend cases.

Do not encode backend-private parameters into `FNACCKernelABI`. Report them via
the private-argument contract so the stable ABI remains usable by Triton,
direct PTX, CUDA Tile IR, and future backends.

### Change a runtime ABI

Update these as one atomic change:

- runtime-call creation in `FNACCLowerToRuntime.cpp`;
- exported runtime function signature;
- JSON parameter roles/types when relevant;
- driver PTX/image ABI validation;
- compatibility aliases or a schema/ABI version;
- MLIR lowering tests and executable integration tests.

Never infer a CUDA pointer's owning device/context from a process-global cache.
All CUDA-owned objects must stay in the state keyed by the active `CUcontext`.

## Diagnostics and troubleshooting

### `FNACC cannot plan launch`

The region is outside the recognised subset. Read the final recognition reason
first; common causes are a noncanonical loop, unsupported expression or call,
too many arrays/scalars, mixed types, non-rank-one reduction access, unconsumed
side effect, or unsupported mutable scalar.

### `FNACC backend selection failed`

The preferred backend is unregistered or its `querySupport` rejected the plan.
Use `--fnacc-backend triton`, allow a Triton fallback, or inspect the detailed
rejection reason. `--fnacc-no-backend-fallback` is useful in tests that must
prove a particular backend handled every kernel.

### Driver fails at `fnacc-pipeline`

Re-run with `--fnacc-verbose --fnacc-keep`, then execute the printed `fir-opt`
command directly. Inspect the `.fir`, `.kernels.ttir`, `.kernels.json`, and
`.host.fir` files in the retained work directory.

### CUDA context initialization errors

The runtime must call `cuInit(0)` before querying the current context. Ensure
the executable is linked against the rebuilt `FortranFNACCRuntime`, not an old
copy found earlier in a library or runtime search path. With
`FNACC_USE_CURRENT_CONTEXT=1`, ensure the caller has already made a valid
context current.

### `update host has no cached allocation`

The object was launched with host-temporary placement or never entered a
persistent lifetime. Use `enter data copyin/create`, `update device`, or
`pack(...:device)` before requesting an explicit host update.

### Descriptor-size fallback warnings

Warnings that object size could not be determined mean lowering used the
raw-pointer compatibility call. That call can update only an allocation that
already exists in the runtime cache. Prefer a supported explicit-shape or
contiguous descriptor form, and treat a `create` sizing failure as a real
compile-time error.

### Multiple embedded bundles

The current executable contract permits one bundle. Consolidate FnACC kernels
into one source/module or into one bundle-producing build step. Do not disable
the driver's check: the strong bundle marker and runtime registration guard
exist to prevent kernel-ID collisions and startup aborts.

### Wrong results with device placement

Check the lifetime in this order:

1. Was the input copied with `copyin` or `update device`?
2. Did host code modify it after that copy?
3. Are all in-place read/write arguments bound to the same cached object?
4. Was `wait` used at an external interoperability boundary?
5. Was the result copied with `update host` or `copyout` before validation?
6. Was the allocation released only after its final consumer?

Enable `FNACC_DEBUG=1` and confirm cache hits, pack targets, byte counts,
extents, grid, tile, CUDA block size, context, and image metadata.

## Known limitations

- FnACC is experimental.
- Only canonical loop nests and recognised expression/reduction/matmul shapes
  are accepted.
- Kernel computation supports ranks one and two; reductions are rank one.
- Data-descriptor operations support only ranks one through three and require
  contiguous storage.
- Device extents use a signed 32-bit runtime ABI.
- One output array, at most three read arrays, and at most three scalar captures
  are supported by the generic expression ABI.
- One reduction scalar/operator is supported per launch.
- Matrix multiplication supports only f32 and f64.
- Triton is the only complete code-generation backend and the normal driver
  emits PTX. Cubin is understood by the typed runtime image ABI but is not the
  normal Triton driver output.
- Mixed backends in one device module are not supported.
- One embedded FnACC bundle is allowed per executable.
- The runtime owns one stream per CUDA context and serialises public entry
  points with a process-wide mutex.
- No source-level `private` clause exists; only provably iteration-private
  scalar temporaries are promoted automatically.
- No asynchronous queue IDs, stream clauses, events exposed to Fortran, or
  cross-stream dependency clauses exist yet.
- Noncontiguous sections, complex/logical/character/derived-type arrays,
  unsigned arithmetic, arbitrary function calls, and general control flow are
  outside the current kernel subset.
- Floating-point min/max, comparisons, transcendental functions, contraction,
  and reduction order inherit backend behaviour; validate NaNs, infinities,
  signed zero, and reproducibility if an application depends on those cases.
