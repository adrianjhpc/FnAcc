# FNACC Documentation

FNACC is an experimental Flang extension for recognising selected Fortran loop
kernels and lowering them through a Triton/CUDA path.

The feature is currently split into two layers:

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

With PACK clauses:

!$fnacc parallel tile(128) pack(a:device, c:device)
do i = 1, n
  c(i) = a(i) + b(i)
end do

Supported pack targets:

host
device

Examples:

pack(a:host)
pack(a:device)
pack(a:device, c:device)

Standalone data-management directives

!$fnacc update host(c)
!$fnacc update device(a)
!$fnacc release(a, c)
!$fnacc release all

or:

!$fnacc release_all

Example: vector addition

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

Current recognised kernel patterns

The compiler currently recognises a limited set of elementwise loop shapes.
1-D binary array expression

c(i) = a(i) + b(i)
c(i) = a(i) - b(i)
c(i) = a(i) * b(i)
c(i) = a(i) / b(i)

SAXPY-style expression

c(i) = alpha * a(i) + b(i)

Generic 1-D expression tree

Examples:

c(i) = alpha * a(i) + beta * b(i)
c(i) = (a(i) + b(i)) * alpha
c(i) = a(i) + 1.0

Subject to current implementation limits on the number of arrays/scalars.
2-D binary array expression

c(i, j) = a(i, j) + b(i, j)

with loop shape:

do j = 1, m
  do i = 1, n
    c(i, j) = a(i, j) + b(i, j)
  end do
end do

Important restrictions
Loop form

Currently supported loops are expected to be canonical Fortran loops:

do i = 1, n

The lower bound must currently be 1, and the step must currently be 1.

Unsupported examples:

do i = 0, n

do i = 1, n, 2

Type support

The prototype primarily supports:

real

which is lowered as f32.
Array layout

The current generated indexing assumes normal Fortran column-major storage.
Runtime cache semantics

Data directives such as:

!$fnacc update device(a)

operate on runtime cache entries. If a variable has never been seen by a previous FNACC launch or allocation path, the runtime may not know its size and may ignore the update.

This is a known limitation of the current pointer-only runtime ABI
