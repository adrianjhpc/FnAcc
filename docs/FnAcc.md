# FNACC Documentation

FNACC is an experimental Flang extension for recognising selected Fortran loop
kernels and lowering them through an accelerator backend. The current prototype
uses a Triton/CUDA path, but the name is intended to be broader than GPU-only
offload.

The feature is split into two layers:

1. **Compiler-side support inside Flang**
   - parses `!$fnacc` directives;
   - constructs FNACC parse-tree nodes;
   - lowers to `fnacc.*` FIR/MLIR operations;
   - emits Triton TTIR and JSON kernel metadata;
   - lowers host-side operations to runtime calls.

2. **External wrapper support**
   - invokes Triton tools;
   - translates TTIR to PTX;
   - compiles transformed host FIR;
   - links the FNACC runtime;
   - sets runtime environment variables.

This repository provides layer 2.

---

## Supported directives

### Parallel loop directive

```fortran
!$fnacc parallel tile(128)
do i = 1, n
  c(i) = a(i) + b(i)
end do
```

### Parallel loop directive with `PACK` clauses

```fortran
!$fnacc parallel tile(128) pack(a:device, c:device)
do i = 1, n
  c(i) = a(i) + b(i)
end do
```

Supported `PACK` targets are:

- `host`
- `device`

Examples:

```fortran
pack(a:host)
pack(a:device)
pack(a:device, c:device)
```

### Standalone data-management directives

```fortran
!$fnacc update host(c)
!$fnacc update device(a)
!$fnacc release(a, c)
!$fnacc release all
```

The following spelling is also accepted:

```fortran
!$fnacc release_all
```

---

## Example: vector addition

```fortran
subroutine compute_add(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$fnacc parallel tile(128) pack(a:device, b:device, c:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  !$fnacc update host(c)
  !$fnacc release all
end subroutine
```

---

## Current recognised kernel patterns

The compiler currently recognises a limited set of elementwise loop shapes.

### 1-D binary array expressions

```fortran
c(i) = a(i) + b(i)
c(i) = a(i) - b(i)
c(i) = a(i) * b(i)
c(i) = a(i) / b(i)
```

### SAXPY-style expressions

```fortran
c(i) = alpha * a(i) + b(i)
```

### Generic 1-D expression trees

Examples:

```fortran
c(i) = alpha * a(i) + beta * b(i)
c(i) = (a(i) + b(i)) * alpha
c(i) = a(i) + 1.0
```

This is subject to current implementation limits on the number of arrays and
scalars in the expression.

### 2-D binary array expressions

```fortran
c(i, j) = a(i, j) + b(i, j)
```

Expected loop shape:

```fortran
do j = 1, m
  do i = 1, n
    c(i, j) = a(i, j) + b(i, j)
  end do
end do
```

---

## Important restrictions

### Loop form

Currently supported loops are expected to be canonical Fortran loops of the
form:

```fortran
do i = 1, n
```

The lower bound must currently be `1`, and the step must currently be `1`.

Unsupported examples:

```fortran
do i = 0, n
```

```fortran
do i = 1, n, 2
```

### Type support

The prototype primarily supports:

```fortran
real
```

which is lowered as `f32`.

### Array layout

The current generated indexing assumes normal Fortran column-major storage.

### Runtime cache semantics

Data directives such as:

```fortran
!$fnacc update device(a)
```

operate on runtime cache entries. If a variable has never been seen by a
previous FNACC launch or allocation path, the runtime may not know its size and
may ignore the update.

This is a known limitation of the current pointer-only runtime ABI.

---

## Compilation pipeline

The wrapper currently performs the following high-level pipeline:

1. Compile Fortran source to FIR.
2. Run the combined FNACC `fir-opt` pipeline.
3. Lower emitted Triton TTIR to Triton GPU IR.
4. Lower Triton GPU IR to LLVM MLIR.
5. Translate LLVM MLIR to LLVM IR.
6. Compile LLVM IR to PTX.
7. Compile transformed host FIR to LLVM IR.
8. Compile host LLVM IR to an object file.
9. Compile the Fortran main program.
10. Link the final executable against the FNACC runtime and CUDA driver.

The key FNACC compiler step is:

```bash
fir-opt \
  --fnacc-pipeline="ttir-output=fnacc_kernels.ttir json-output=fnacc_kernels.json emit-fortran-aliases=true" \
  input.fir \
  -o input.host.fir
```

---

## Runtime environment variables

The current runtime uses external files for kernel code and metadata.

### `FNACC_PTX`

Path to the generated PTX file.

```bash
export FNACC_PTX=fnacc_kernels.ptx
```

### `FNACC_KERNELS_JSON`

Path to the generated JSON kernel metadata.

```bash
export FNACC_KERNELS_JSON=fnacc_kernels.json
```

### `FNACC_DEBUG`

Enable runtime debug logging.

```bash
export FNACC_DEBUG=1
```

---

## Current limitations

FNACC is experimental. Current limitations include:

- only selected elementwise loop kernels are recognised;
- `real` / `f32` is the primary supported numeric type;
- loop lower bounds must currently be `1`;
- loop steps must currently be `1`;
- only a small number of read arrays and scalar values are supported;
- standalone data directives currently rely on runtime cache state;
- the runtime ABI is currently pointer-based and does not yet carry full
  Fortran descriptor information;
- the current backend path targets Triton/CUDA/PTX;
- Triton invocation is handled by an external wrapper rather than directly by
  Flang.

---

## Recommended usage

Use the external wrapper script from this repository rather than invoking each
tool manually.

Example:

```bash
./bin/fnacc-flang \
  --kernel-src examples/vector-add/kernel.f90 \
  --main-src examples/vector-add/main.f90 \
  --workdir build/vector-add \
  --base vector-add \
  -o build/vector-add/vector-add
```

Run the generated launcher:

```bash
./build/vector-add/vector-add.run
```

Enable runtime debug output:

```bash
FNACC_DEBUG=1 ./build/vector-add/vector-add.run
```

