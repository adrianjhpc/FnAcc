# FnACC 

FnACC is a programming approach (think OpenACC or OpenMP) to enable porting Fortran programs to GPUs (or other accelerators). It is designed to leverage [Triton](https://triton-lang.org/main/index.html) to port to the accelerator(s).

## Examples and Triton Binding Wrapper

This repository contains examples and a wrapper script for the experimental Flang FnACC extension.

FnACC adds prototype Fortran directives such as:

```fortran
!$fnacc parallel tile(128) pack(a:device, c:device)
do i = 1, n
  c(i) = a(i) + b(i)
end do
